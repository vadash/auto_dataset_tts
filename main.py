import glob
import os
import randomname
import librosa
import numpy as np
import soundfile as sf
import whisperx
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from typing import List
import logging
import torch

# --- Constants ---
DEVICE = "cuda"  # Device for model execution
BATCH_SIZE = 16  # Batch size for processing (adjust based on VRAM)
MODEL_NAME = "small.en"  # Model name (large-v2 or small.en)
COMPUTE_TYPE = "int8"  # Computation type (float16 or int8)
LANGUAGE = "en"  # Language for transcription
TARGET_SAMPLING_RATE = 44100  # Target sampling rate for audio
LONG_CHAR_THRESHOLD = 0.5  # Threshold to identify long pauses in speech
FRAME_LENGTH_MS = 25  # Frame length in milliseconds for energy calculation
HOP_LENGTH_MS = 10  # Hop length in milliseconds for energy calculation
SMOOTHING_WINDOW_SIZE = 8  # Increased from 5 to 8 for smoother detection
STD_RMS_MULTIPLIER_HIGH = 1.0  # To avoid detecting short, intentional pauses within speech as silence.
STD_RMS_MULTIPLIER_LOW = 0.25  # To ensure that only genuine periods of silence are detected, not just brief dips in volume.
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds
TRAIN_VALIDATION_SPLIT = 0.90  # Percentage of data to use for training
# RVC use 3,4,5. XTTS use 8,10,12
MIN_SEGMENT_DURATION = 3  # Minimum duration of a segment in seconds
OPTIMAL_SEGMENT_DURATION = 4  # Optimal duration of a segment in seconds
MAX_SEGMENT_DURATION = 5  # Maximum duration of a segment in seconds

# --- Logging Setup ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the minimum logging level

# Create a file handler to write logs to a file
log_file = 'audio_processing.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)  # Set the minimum logging level for the file handler

# Create a console handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the minimum logging level for the console handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Model Initialization ---
logger.info("Initializing WhisperX and Alignment models...")
try:
    model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
    logger.info("WhisperX and Alignment models initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

# --- Run Management ---
def get_run_name() -> str:
    """Generate a unique run name using the current timestamp and a random name."""
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{randomname.get_name()}"
    logger.info(f"Starting a new run: {run_name}")
    return run_name

run_name = get_run_name()
output_dir = os.path.join('output_audio_segments', run_name)
os.makedirs(output_dir, exist_ok=True)

# --- Track processed files ---
files_done_path = os.path.join(output_dir, 'files_done.txt')
files_done = set()
if os.path.exists(files_done_path):
    with open(files_done_path, 'r', encoding='utf-8') as f:
        files_done = set(x.strip() for x in f.readlines())

# --- Data Structures ---
@dataclass
class Segment:
    """Dataclass to store segment information."""
    text: str
    filepath: str
    duration: float

# --- Dynamic Chunk Size Adjustment ---
def calculate_dynamic_chunk_size(audio: np.ndarray, sr: int) -> int:
    """
    Calculates a dynamic chunk size based on audio characteristics.
    """
    # Example: Adjust chunk size based on average silence duration
    rms = librosa.feature.rms(y=audio, frame_length=int(sr * FRAME_LENGTH_MS / 1000), hop_length=int(sr * HOP_LENGTH_MS / 1000))[0]
    avg_silence_duration = np.mean(np.diff(np.where(rms < np.mean(rms))[0])) * HOP_LENGTH_MS / 1000
    dynamic_chunk_size = max(10, min(30, int(avg_silence_duration * 5)))  # Example heuristic
    logger.debug(f"Dynamic chunk size: {dynamic_chunk_size}")
    return dynamic_chunk_size

# --- Enhanced Silence Detection ---
def detect_silence_with_hysteresis(audio: np.ndarray, sr: int, silence_threshold_high: float, silence_threshold_low: float, min_silence_frames: int) -> List[tuple]:
    """
    Detects periods of silence in audio using hysteresis thresholding.
    """
    rms = librosa.feature.rms(y=audio, frame_length=int(sr * FRAME_LENGTH_MS / 1000), hop_length=int(sr * HOP_LENGTH_MS / 1000))[0]
    smoothed_rms = np.convolve(rms, np.ones(SMOOTHING_WINDOW_SIZE) / SMOOTHING_WINDOW_SIZE, mode='same')

    silence_starts = []
    silence_ends = []
    in_silence = False
    silence_frame_count = 0

    for i, level in enumerate(smoothed_rms):
        if level < silence_threshold_high:
            silence_frame_count += 1
            if not in_silence and silence_frame_count >= min_silence_frames:
                in_silence = True
                silence_starts.append(max(0, i - min_silence_frames))
        else:
            if in_silence and all(smoothed_rms[i - j] < silence_threshold_low for j in range(min_silence_frames)):
                in_silence = False
                silence_ends.append(i)
            silence_frame_count = 0

    # Handle the case where the audio ends in silence
    if in_silence:
        silence_ends.append(len(smoothed_rms))

    return list(zip(silence_starts, silence_ends))

# --- Segment Processing ---
def process_segments(result, audio_path, sr):
    """
    Processes and combines segments to meet length requirements while preserving sentence integrity.
    Returns a list of refined segments.
    """
    refined_segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None
    }
    min_valid_duration = 0.5  # Minimum valid segment duration in seconds

    # Calculate silence segments
    audio = whisperx.load_audio(audio_path)
    median_rms = np.median(librosa.feature.rms(y=audio, frame_length=int(sr * FRAME_LENGTH_MS / 1000), hop_length=int(sr * HOP_LENGTH_MS / 1000))[0])
    std_rms = np.std(librosa.feature.rms(y=audio, frame_length=int(sr * FRAME_LENGTH_MS / 1000), hop_length=int(sr * HOP_LENGTH_MS / 1000))[0])
    silence_threshold_high = median_rms + STD_RMS_MULTIPLIER_HIGH * std_rms
    silence_threshold_low = median_rms + STD_RMS_MULTIPLIER_LOW * std_rms
    min_silence_frames = int(MIN_SILENCE_DURATION / (HOP_LENGTH_MS / 1000))
    silence_segments = detect_silence_with_hysteresis(audio, sr, silence_threshold_high, silence_threshold_low, min_silence_frames)

    for segment in result["segments"]:
        # Skip segments that are too short
        if segment['end'] - segment['start'] < min_valid_duration:
            continue

        if current_segment['start'] is None:
            current_segment['start'] = segment['start']

        segment_duration = segment['end'] - segment['start']
        proposed_duration = segment['end'] - current_segment['start']

        # Check if adding this segment would exceed the maximum duration
        if proposed_duration > MAX_SEGMENT_DURATION:
            # Save the current segment if it meets the minimum duration
            if current_segment['end'] is not None and current_segment['end'] - current_segment['start'] >= MIN_SEGMENT_DURATION:
                refined_segments.append(current_segment)
            # Start a new segment
            current_segment = {
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end']
            }
            continue

        # Add text to the current segment
        if current_segment['text']:
            current_segment['text'] += ' ' + segment['text']
        else:
            current_segment['text'] = segment['text']

        current_segment['end'] = segment['end']

        # Check if we should close this segment based on silence or punctuation
        close_segment = False
        if proposed_duration >= OPTIMAL_SEGMENT_DURATION:
            # Check for silence at the end of the segment
            segment_end_frame = int(segment['end'] * sr / (sr * HOP_LENGTH_MS / 1000))
            if any(start <= segment_end_frame <= end for start, end in silence_segments):
                close_segment = True
            # Check for punctuation at the end of the segment
            elif any(segment['text'].strip().endswith(p) for p in ['.', '!', '?', ':', ';']):
                close_segment = True

        if close_segment:
            # Save the current segment if it meets the minimum duration
            if current_segment['end'] is not None and current_segment['end'] - current_segment['start'] >= MIN_SEGMENT_DURATION:
                refined_segments.append(current_segment)
                current_segment = {
                    'text': '',
                    'start': None,
                    'end': None
                }

    # Add the last segment if it meets the minimum duration
    if current_segment['start'] is not None and current_segment['end'] is not None:
        final_duration = current_segment['end'] - current_segment['start']
        if final_duration >= MIN_SEGMENT_DURATION:
            current_segment['end'] += 0.1  # Add 100ms buffer to the last segment
            refined_segments.append(current_segment)

    return refined_segments

# --- Audio Trimming ---
def cut_sample_to_speech_only(audio_path: str, target_sampling_rate: int) -> str:
    """
    Transcribes an audio sample, finds the end of the last speech segment,
    and trims the audio to that point to remove trailing silence or noise.
    Uses adaptive silence thresholding and smoothed energy for improved accuracy.
    """
    try:
        logger.debug(f"Processing audio: {audio_path}")
        audio_sample = whisperx.load_audio(audio_path)
        dynamic_chunk_size = calculate_dynamic_chunk_size(audio_sample, target_sampling_rate)
        result_sample = model.transcribe(audio_sample, batch_size=BATCH_SIZE, chunk_size=dynamic_chunk_size)
        result_sample = whisperx.align(result_sample["segments"], model_a, metadata, audio_sample, DEVICE, return_char_alignments=True)

        if not result_sample['segments']:
            logger.warning(f"No speech segments found in {audio_path}.")
            return ""

        end_buffer_time = 0.2  # Add 200ms buffer after last detected speech
        min_valid_segments = 3  # Minimum number of segments to check

        # Get multiple end points and average them
        if len(result_sample['segments']) >= min_valid_segments:
            last_segments = result_sample['segments'][-min_valid_segments:]
            end_times = []

            for segment in last_segments:
                if segment.get('chars'):
                    chars = [char for char in segment['chars']
                             if char.get('end') is not None and char.get('start') is not None]
                    if chars:
                        valid_end_chars = [char['end'] for char in reversed(chars)
                                           if char.get('char', '').strip() and
                                           (char['end'] - char['start']) < LONG_CHAR_THRESHOLD]
                        if valid_end_chars:
                            end_times.append(valid_end_chars[0])

            if end_times:
                end = max(end_times) + end_buffer_time
            else:
                end = last_segments[-1]['end'] + end_buffer_time
        else:
            end = result_sample['segments'][-1]['end'] + end_buffer_time

        audio, sr = librosa.load(audio_path, sr=target_sampling_rate)
        frame_length = int(FRAME_LENGTH_MS / 1000 * sr)
        hop_length = int(HOP_LENGTH_MS / 1000 * sr)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        smoothed_rms = np.convolve(rms, np.ones(SMOOTHING_WINDOW_SIZE) / SMOOTHING_WINDOW_SIZE, mode='same')

        median_rms = np.median(smoothed_rms)
        std_rms = np.std(smoothed_rms)
        silence_threshold_high = median_rms + STD_RMS_MULTIPLIER_HIGH * std_rms
        silence_threshold_low = median_rms + STD_RMS_MULTIPLIER_LOW * std_rms
        required_silence_frames = int(0.4 / (hop_length / sr))  # 400ms of consistent silence

        end_frame = int(end / (hop_length / sr))

        # Look for a longer period of consistent silence
        for i in range(end_frame, min(len(smoothed_rms), end_frame + int(1.0 * sr / hop_length))):
            if smoothed_rms[i] < silence_threshold_high:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= required_silence_frames:
                # Verify silence is consistent
                if all(smoothed_rms[i - j] < silence_threshold_low for j in range(required_silence_frames)):
                    end = (i - required_silence_frames // 2) * (hop_length / sr)
                    break

        end += 0.1  # Add 100ms buffer

        sf.write(audio_path, audio[:int(end * sr)], sr)
        logger.debug(f"Trimmed audio saved: {audio_path}")
        return " ".join(seg['text'].strip() for seg in result_sample['segments'])

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}", exc_info=True)
        return ""

# --- Audio Cutting and Saving ---
def cut_and_save_audio(input_audio_path: str, segments: List[dict]) -> List[Segment]:
    """
    Loads audio, cuts segments based on start and end times, and saves them.
    """
    output_dir_name = os.path.join(output_dir, 'audio')
    os.makedirs(output_dir_name, exist_ok=True)

    try:
        audio, original_sampling_rate = librosa.load(input_audio_path, sr=TARGET_SAMPLING_RATE)
    except Exception as e:
        logger.error(f"Failed to load audio file {input_audio_path}: {e}", exc_info=True)
        return []

    outputs = []
    output_prefix = os.path.splitext(os.path.basename(input_audio_path))[0]

    for idx, segment in tqdm(enumerate(segments), desc=f"Cutting {input_audio_path}", total=len(segments), leave=False):
        start_sample = int(segment['start'] * original_sampling_rate)
        end_sample = int(segment['end'] * original_sampling_rate)
        audio_segment = audio[start_sample:end_sample]
        output_path = os.path.join(output_dir_name, f"{output_prefix}_{idx + 1}.wav")
        sf.write(output_path, audio_segment, TARGET_SAMPLING_RATE)
        logger.debug(f"Saved audio segment: {output_path}")

        segment_text = cut_sample_to_speech_only(output_path, TARGET_SAMPLING_RATE)
        if segment_text:
            outputs.append(Segment(
                text=segment_text,
                filepath=output_path,
                duration=librosa.get_duration(y=audio_segment, sr=TARGET_SAMPLING_RATE)
            ))
        else:
            logger.warning(f"Removing segment file due to processing error: {output_path}")
            try:
                os.remove(output_path)
            except Exception as e:
                logger.error(f"Error deleting file '{output_path}': {e}", exc_info=True)

    return outputs

# --- Main Processing Function ---
def create_segments_for_files(files_to_segment: List[str]):
    """
    Transcribes, aligns, and segments audio files sequentially, then saves the segments.
    """
    for audio_file in tqdm(sorted(files_to_segment, key=os.path.getsize, reverse=True), desc="Processing files"):
        if audio_file in files_done:
            logger.info(f"Skipping already processed file: {audio_file}")
            continue

        try:
            logger.info(f"Processing file: {audio_file}")
            audio = whisperx.load_audio(audio_file)
            dynamic_chunk_size = calculate_dynamic_chunk_size(audio, TARGET_SAMPLING_RATE)
            result = model.transcribe(audio, batch_size=BATCH_SIZE, chunk_size=dynamic_chunk_size)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=True)

            refined_segments = process_segments(result, audio_file, TARGET_SAMPLING_RATE)
            segment_objects = cut_and_save_audio(audio_file, refined_segments)

            if not segment_objects:
                logger.warning(f"No segments created for {audio_file}. Skipping file.")
                continue

            split_id = int(TRAIN_VALIDATION_SPLIT * len(segment_objects))
            train_segments = segment_objects[:split_id]
            validation_segments = segment_objects[split_id:]

            with open(os.path.join(output_dir, 'train.txt'), 'a', encoding='utf-8') as f:
                for segment in train_segments:
                    f.write(f"{segment.filepath}|{segment.text.strip()}\n")
            logger.debug(f"Wrote {len(train_segments)} segments to train.txt")

            with open(os.path.join(output_dir, 'validation.txt'), 'a', encoding='utf-8') as f:
                for segment in validation_segments:
                    f.write(f"{segment.filepath}|{segment.text.strip()}\n")
            logger.debug(f"Wrote {len(validation_segments)} segments to validation.txt")

            with open(files_done_path, 'a', encoding='utf-8') as f:
                f.write(f"{audio_file}\n")
            logger.info(f"Finished processing: {audio_file}")

        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == '__main__':
    logger.info(f"Starting run for: {run_name}")
    files_to_segment = glob.glob("full_audio_files/*.wav")
    if not files_to_segment:
        logger.warning("No .wav files found in the 'full_audio_files' directory.")
    else:
        create_segments_for_files(files_to_segment)
    logger.info("Audio processing complete.")
