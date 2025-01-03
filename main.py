import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import librosa
import numpy as np
import randomname
import soundfile as sf
import whisperx
from tqdm import tqdm

# Configuration
device = "cuda"
batch_size = 8  # Use 8 for 8 GB VRAM
compute_type = "float16"  # float16 / int8
language = "en"
target_sampling_rate = 44100  # Default and most popular WAV format sampling rate

# Model Initialization
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language)
model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

# Run Management
def get_run_name() -> str:
    """Generate a unique run name using the current timestamp and a random name."""
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{randomname.get_name()}"
    print(f"Starting a new run: {run_name}")
    return run_name

run_name = get_run_name()
output_dir = os.path.join('output_audio_segments', run_name)
os.makedirs(output_dir, exist_ok=True)

# Track processed files
files_done_path = os.path.join(output_dir, 'files_done.txt')
files_done = set()
if os.path.exists(files_done_path):
    with open(files_done_path, 'r', encoding='utf-8') as f:
        files_done = set(x.strip() for x in f.readlines())

@dataclass
class Segment:
    """Dataclass to store segment information."""
    text: str
    filepath: str
    duration: float

def process_segments(result):
    """
    Processes and combines segments to meet length requirements while preserving sentence integrity.
    Returns list of refined segments.
    """
    refined_segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None
    }

    for segment in result["segments"]:
        # Initialize start time if this is the beginning of a new segment
        if current_segment['start'] is None:
            current_segment['start'] = segment['start']

        # Calculate current duration
        segment_duration = segment['end'] - current_segment['start']
        proposed_duration = segment['end'] - current_segment['start']

        # Check if adding this segment would exceed max duration
        if proposed_duration > 15:
            if segment_duration >= 3:  # Only save if minimum duration met
                refined_segments.append(current_segment)
                current_segment = {
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                }
            continue

        # Add text to current segment
        if current_segment['text']:
            current_segment['text'] += ' ' + segment['text']
        else:
            current_segment['text'] = segment['text']

        current_segment['end'] = segment['end']

        # Check if we should close this segment (near optimal length or ends with punctuation)
        if (segment_duration >= 8 and
            any(segment['text'].strip().endswith(p) for p in ['.', '!', '?', ':', ';'])):
            if segment_duration >= 3:  # Only save if minimum duration met
                refined_segments.append(current_segment)
                current_segment = {
                    'text': '',
                    'start': None,
                    'end': None
                }

    # Add the last segment if it meets minimum duration
    if current_segment['start'] is not None:
        final_duration = current_segment['end'] - current_segment['start']
        if final_duration >= 3:
            refined_segments.append(current_segment)

    return refined_segments

def cut_sample_to_speech_only(audio_path: str) -> str:
    """
    Transcribes an audio sample, finds the end of the last speech segment,
    and trims the audio to that point to remove trailing silence or noise.
    """
    try:
        audio_sample = whisperx.load_audio(audio_path)
        result_sample = model.transcribe(audio_sample, batch_size=batch_size, chunk_size=30)
        result_sample = whisperx.align(result_sample["segments"], model_a, metadata, audio_sample, device, return_char_alignments=True)

        # Find the end of the last speech segment
        segment = result_sample['segments'][-1]
        chars = [char for char in segment['chars'] if char.get('end')]
        long_char_threshold = 0.5  # Threshold to identify long pauses
        end = next((char['end'] for char in reversed(chars) if char.get('char').strip() and (char['end'] - char['start']) < long_char_threshold), chars[-1]['end'])

        # Refine end time based on silence detection
        audio, sr = librosa.load(audio_path, sr=target_sampling_rate)
        amplitude_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        window_size_sec = 0.10
        silence_threshold_db = -40  # Default audiobook silence threshold
        window_size_samples = int(window_size_sec * sr)
        start_frame = int(end * sr)
        audio_windows = [amplitude_db[:, x:x + window_size_samples].max() for x in range(start_frame, amplitude_db.shape[1], window_size_samples)]

        for window in audio_windows:
            if window > silence_threshold_db:
                end += window_size_sec
            else:
                break

        # Save the trimmed audio
        sf.write(audio_path, audio[:int(end * sr)], sr)
        return " ".join(seg['text'].strip() for seg in result_sample['segments'])

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return ""

def cut_and_save_audio(input_audio_path: str, segments: List[dict]) -> List[Segment]:
    """
    Loads audio, cuts segments based on start and end times, and saves them.
    """
    output_dir_name = os.path.join(output_dir, 'audio')
    os.makedirs(output_dir_name, exist_ok=True)

    audio, original_sampling_rate = librosa.load(input_audio_path, sr=target_sampling_rate)
    outputs = []
    output_prefix = os.path.splitext(os.path.basename(input_audio_path))[0]

    for idx, segment in tqdm(enumerate(segments), desc=input_audio_path, total=len(segments), leave=False):
        start_sample = int(segment['start'] * original_sampling_rate)
        end_sample = int(segment['end'] * original_sampling_rate)
        audio_segment = audio[start_sample:end_sample]
        output_path = os.path.join(output_dir_name, f"{output_prefix}_{idx + 1}.wav")
        sf.write(output_path, audio_segment, target_sampling_rate)

        segment_text = cut_sample_to_speech_only(output_path)
        if segment_text:
            outputs.append(Segment(
                text=segment_text,
                filepath=output_path,
                duration=librosa.get_duration(y=audio_segment, sr=target_sampling_rate)
            ))
        else:
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"Error deleting file '{output_path}': {e}")

    return outputs

def create_segments_for_files(files_to_segment: List[str]):
    """
    Transcribes, aligns, and segments audio files, then saves the segments.
    """
    for audio_file in tqdm(sorted(files_to_segment, key=os.path.getsize, reverse=True), desc="Processing files"):
        if audio_file in files_done:
            continue

        try:
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size, chunk_size=30)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

            # Process segments to meet our criteria
            refined_segments = process_segments(result)

            # Continue with the rest of the processing using refined_segments
            segment_objects = cut_and_save_audio(audio_file, refined_segments)

            # Save segments to train and validation files
            split_id = int(0.95 * len(segment_objects))
            train_segments = segment_objects[:split_id]
            validation_segments = segment_objects[split_id:]

            with open(os.path.join(output_dir, 'train.txt'), 'a', encoding='utf-8') as f:
                for segment in train_segments:
                    f.write(f"{segment.filepath}|{segment.text.strip()}\n")

            with open(os.path.join(output_dir, 'validation.txt'), 'a', encoding='utf-8') as f:
                for segment in validation_segments:
                    f.write(f"{segment.filepath}|{segment.text.strip()}\n")

            # Mark file as processed
            with open(files_done_path, 'a', encoding='utf-8') as f:
                f.write(f"{audio_file}\n")

        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")

if __name__ == '__main__':
    print(f"Starting run for: {run_name}")
    files_to_segment = glob.glob("full_audio_files/*.wav")
    create_segments_for_files(files_to_segment)
