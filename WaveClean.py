import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment, silence
from pydub.effects import normalize, compress_dynamic_range
from scipy import signal
import sys
import os
import tempfile
import subprocess
import platform

def get_script_dir():
    """Get the directory where the script is located"""
    return os.path.dirname(os.path.abspath(__file__))

def find_ffmpeg():
    """Find ffmpeg executable - first in script directory, then in PATH"""
    script_dir = get_script_dir()
    
    # Check for ffmpeg in script directory
    ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    local_ffmpeg = os.path.join(script_dir, ffmpeg_name)
    
    if os.path.isfile(local_ffmpeg):
        return local_ffmpeg
    
    # Check if ffmpeg is in PATH
    try:
        if platform.system() == "Windows":
            subprocess.check_output('where ffmpeg', shell=True)
        else:
            subprocess.check_output('which ffmpeg', shell=True)
        return "ffmpeg"  # return just the command if it's in PATH
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def find_ffprobe():
    """Find ffprobe executable - first in script directory, then in PATH, then try to infer from ffmpeg location"""
    script_dir = get_script_dir()
    
    # Check for ffprobe in script directory
    ffprobe_name = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"
    local_ffprobe = os.path.join(script_dir, ffprobe_name)
    
    if os.path.isfile(local_ffprobe):
        return local_ffprobe
    
    # Check if ffprobe is in PATH
    try:
        if platform.system() == "Windows":
            subprocess.check_output('where ffprobe', shell=True)
        else:
            subprocess.check_output('which ffprobe', shell=True)
        return "ffprobe"  # return just the command if it's in PATH
    except (subprocess.SubprocessError, FileNotFoundError):
        # If we can't find ffprobe directly, try to infer from ffmpeg location
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path and ffmpeg_path != "ffmpeg":
            # If we found ffmpeg as a full path, check if ffprobe is in the same directory
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            possible_ffprobe = os.path.join(ffmpeg_dir, ffprobe_name)
            if os.path.isfile(possible_ffprobe):
                return possible_ffprobe
            
            # Explicitly check the 'bin' folder inside the FFmpeg directory
            bin_ffprobe = os.path.join(ffmpeg_dir, "bin", ffprobe_name)
            if os.path.isfile(bin_ffprobe):
                return bin_ffprobe
        return None

def print_installation_instructions():
    """Print installation instructions for dependencies"""
    script_dir = get_script_dir()
    
    print("\n=== WaveClean Installation Instructions ===")
    
    # FFmpeg installation instructions
    print("\nFFmpeg Installation:")
    if platform.system() == "Windows":
        print(f"1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/")
        print(f"   - The 'ffmpeg-git-full.7z' package is recommended")
        print(f"2. Extract the archive file")
        print(f"3. Navigate to the 'bin' folder in the extracted directory")
        print(f"4. Copy BOTH ffmpeg.exe AND ffprobe.exe to this directory: {script_dir}")
        print("   OR add the bin folder to your PATH environment variable")
    elif platform.system() == "Darwin":  # macOS
        print("1. Install via Homebrew: brew install ffmpeg")
    else:  # Linux
        print("1. Install via package manager: sudo apt-get install ffmpeg")
    
    print("\nAfter installing the dependencies, run the script again.\n")

def is_mp4_file(file_path):
    """Check if the file is an MP4 video file"""
    _, ext = os.path.splitext(file_path.lower())
    return ext == '.mp4'

def extract_audio_from_mp4(mp4_file, ffmpeg_path):
    """Extract audio from MP4 file into a temporary WAV file using ffmpeg directly"""
    temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    try:
        # Use ffmpeg to extract audio from mp4 to wav
        subprocess.run([
            ffmpeg_path,
            '-i', mp4_file,
            '-q:a', '0',         # Best quality
            '-map', 'a',         # Extract audio only
            '-y',                # Overwrite output file if it exists
            temp_audio_file
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return temp_audio_file
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
        raise RuntimeError(f"FFmpeg failed to extract audio: {error_msg}")

def _sample_width_to_dtype(sample_width):
    """Map pydub sample_width (bytes) to numpy int dtype and full-scale value."""
    if sample_width == 1:
        return np.int8, 128.0
    if sample_width == 2:
        return np.int16, 32768.0
    if sample_width == 4:
        return np.int32, 2147483648.0
    return np.int16, 32768.0


def process_advanced_audio(input_file, output_file,
                          silence_threshold=-40,
                          min_silence_len=0.3,
                          breath_cutoff=100,
                          target_dBFS=-20.0,
                          noise_reduction_strength=0.5,
                          remove_silence=True,
                          ffmpeg_path="ffmpeg"):

    # Handle MP4 files by extracting audio first
    temp_file = None
    if is_mp4_file(input_file):
        temp_file = extract_audio_from_mp4(input_file, ffmpeg_path)
        input_file = temp_file

    audio_segment = AudioSegment.from_file(input_file)
    sample_rate = audio_segment.frame_rate
    channels = audio_segment.channels
    sample_width = audio_segment.sample_width
    np_dtype, max_val = _sample_width_to_dtype(sample_width)

    # Decode to float, preserving channels (shape: (channels, frames))
    raw = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / max_val
    if channels > 1:
        raw = raw.reshape(-1, channels).T
    else:
        raw = raw[np.newaxis, :]

    # Build the noise profile from quiet sections of a mono mix
    mono_mix = raw.mean(axis=0)
    non_silent_intervals = librosa.effects.split(
        mono_mix,
        top_db=abs(silence_threshold),
        frame_length=2048,
        hop_length=512
    )

    noise_samples = []
    prev_end = 0
    for start, end in non_silent_intervals:
        if start > prev_end:
            seg = mono_mix[prev_end:start]
            if len(seg) > 0:
                noise_samples.append(seg)
        prev_end = end

    if noise_samples:
        noise_signal = np.concatenate(noise_samples)
        if len(noise_signal) < sample_rate:
            reps = (sample_rate // max(len(noise_signal), 1)) + 1
            noise_signal = np.tile(noise_signal, reps)[:sample_rate]
    else:
        noise_signal = mono_mix[:min(sample_rate, len(mono_mix))]

    # Single, gentle spectral denoise per channel (avoids "musical noise" from
    # stacking aggressive passes).
    cleaned = np.empty_like(raw)
    for c in range(raw.shape[0]):
        cleaned[c] = nr.reduce_noise(
            y=raw[c],
            y_noise=noise_signal,
            sr=sample_rate,
            prop_decrease=noise_reduction_strength,
            n_fft=1024,
            hop_length=256,
            stationary=False,
            use_tqdm=False
        )

    # Single high-pass for rumble/breath. Applied once here instead of
    # stacking three high-pass filters across the pipeline.
    if breath_cutoff and breath_cutoff > 0:
        sos = signal.butter(2, breath_cutoff, 'hp', fs=sample_rate, output='sos')
        for c in range(cleaned.shape[0]):
            cleaned[c] = signal.sosfiltfilt(sos, cleaned[c])

    # Re-interleave and clip before quantizing back to int.
    cleaned = np.clip(cleaned, -1.0, 1.0)
    interleaved = (cleaned.T.reshape(-1) * (max_val - 1)).astype(np_dtype)

    processed_audio = AudioSegment(
        interleaved.tobytes(),
        frame_rate=sample_rate,
        sample_width=sample_width,
        channels=channels
    )

    # Gentle compression — preserves dynamics for voice content.
    processed_audio = compress_dynamic_range(
        processed_audio, threshold=-20.0, ratio=2.0, attack=5, release=50
    )

    if remove_silence:
        audio_chunks = silence.split_on_silence(
            processed_audio,
            silence_thresh=silence_threshold,
            min_silence_len=int(min_silence_len * 1000),
            keep_silence=200
        )
        if audio_chunks:
            output = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                output = output.append(chunk, crossfade=20)
        else:
            output = processed_audio
    else:
        output = processed_audio

    output = normalize(output, headroom=1.5)
    if output.dBFS != float('-inf'):
        output = output.apply_gain(target_dBFS - output.dBFS)

    # Pick output format from the file extension instead of forcing MP3.
    _, ext = os.path.splitext(output_file)
    fmt = ext.lstrip('.').lower() or 'wav'
    export_kwargs = {"format": fmt}
    if fmt in ("mp3",):
        export_kwargs["bitrate"] = "192k"
    elif fmt in ("m4a", "aac"):
        export_kwargs["bitrate"] = "256k"

    output.export(output_file, **export_kwargs)

    # Clean up temporary file if one was created
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    # Check for dependencies first
    ffmpeg_path = find_ffmpeg()
    ffprobe_path = find_ffprobe()
    
    # Configure pydub to use the found executables
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        print(f"Using FFmpeg from: {ffmpeg_path}")
    
    if ffprobe_path:
        from pydub.utils import mediainfo_json
        mediainfo_json.ffprobe_path = ffprobe_path
        print(f"Using FFprobe from: {ffprobe_path}")
    
    if len(sys.argv) < 2:
        print("Usage: python WaveClean.py <input_file> [output_file]")
        print("Example: python WaveClean.py input.m4a processed_output.mp3")
        print("         python WaveClean.py input.mp4 processed_output.mp3")
        
        if not ffmpeg_path or not ffprobe_path:
            print_installation_instructions()
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if dependencies are installed
    if not ffmpeg_path:
        print(f"Error: FFmpeg not found in the script directory or system PATH.")
        print_installation_instructions()
        sys.exit(1)
    
    if not ffprobe_path:
        print(f"Error: FFprobe not found in the script directory or system PATH.")
        print(f"Note: FFprobe is typically included with FFmpeg in the same download package.")
        print_installation_instructions()
        sys.exit(1)
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "processed_output.mp3"
    
    try:
        process_advanced_audio(
            input_file,
            output_file,
            silence_threshold=-50,
            min_silence_len=0.2,
            breath_cutoff=100,
            target_dBFS=-16.0,
            ffmpeg_path=ffmpeg_path
        )
        print(f"Audio processing completed successfully! Output saved to: {output_file}")
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        print_installation_instructions()
        sys.exit(1)