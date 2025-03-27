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

def process_advanced_audio(input_file, output_file, 
                          silence_threshold=-40, 
                          min_silence_len=0.3,
                          breath_cutoff=200,
                          target_dBFS=-20.0,
                          ffmpeg_path="ffmpeg"):

    # Handle MP4 files by extracting audio first
    temp_file = None
    if is_mp4_file(input_file):
        temp_file = extract_audio_from_mp4(input_file, ffmpeg_path)
        input_file = temp_file

    audio_segment = AudioSegment.from_file(input_file).set_channels(1)
    sample_rate = audio_segment.frame_rate
    
    audio_segment = audio_segment.high_pass_filter(80)
    
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32767.0

    non_silent_intervals = librosa.effects.split(
        samples,
        top_db=abs(silence_threshold),
        frame_length=2048,
        hop_length=512
    )

    noise_samples = []
    prev_end = 0
    for start, end in non_silent_intervals:
        if start > prev_end:
            noise_segment = samples[prev_end:start]
            if len(noise_segment) > 0:
                noise_samples.append(noise_segment)
        prev_end = end

    if len(noise_samples) > 0:
        noise_signal = np.concatenate(noise_samples)
        if len(noise_signal) < sample_rate:  # If collected noise is less than 1 second
            noise_signal = np.tile(noise_signal, (sample_rate // len(noise_signal)) + 1)[:sample_rate]
    else:
        noise_signal = samples[:sample_rate]

    cleaned_signal1 = nr.reduce_noise(
        y=samples,
        y_noise=noise_signal,
        sr=sample_rate,
        prop_decrease=0.7,
        n_fft=512,
        hop_length=128,
        stationary=False,
        use_tqdm=False
    )

    sos = signal.butter(2, breath_cutoff, 'hp', fs=sample_rate, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, cleaned_signal1)

    # Second pass noise reduction
    cleaned_signal2 = nr.reduce_noise(
        y=filtered_signal,
        y_noise=noise_signal,
        sr=sample_rate,
        prop_decrease=0.5,
        n_fft=1024,
        hop_length=256,
        stationary=True,
        use_tqdm=False
    )

    processed_audio = AudioSegment(
        (cleaned_signal2 * 32767).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

    processed_audio = compress_dynamic_range(processed_audio, threshold=-35, ratio=4.0, attack=5, release=50)

    audio_chunks = silence.split_on_silence(
        processed_audio,
        silence_thresh=silence_threshold,
        min_silence_len=int(min_silence_len * 1000),
        keep_silence=150
    )

    faded_chunks = []
    for chunk in audio_chunks:
        duration = len(chunk)
        fade_time = min(50, duration // 4)  # Maximum 50ms fade
        faded_chunks.append(chunk.fade_in(fade_time).fade_out(fade_time))

    output = sum(faded_chunks, AudioSegment.empty())

    output = normalize(output, headroom=1.5)
    output = output.apply_gain(target_dBFS - output.dBFS)

    output = output.low_pass_filter(10000).high_pass_filter(120)

    output.export(output_file, format="mp3", bitrate="192k")
    
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