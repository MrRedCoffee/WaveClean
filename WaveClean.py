import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment, silence
from pydub.effects import normalize, compress_dynamic_range
from scipy import signal
import sys

def process_advanced_audio(input_file, output_file, 
                          silence_threshold=-40, 
                          min_silence_len=0.3,
                          breath_cutoff=200,
                          target_dBFS=-20.0):

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio.py <input_file> [output_file]")
        print("Example: python audio.py input.m4a processed_output.mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "processed_output.mp3"
    
    process_advanced_audio(
        input_file,
        output_file,
        silence_threshold=-50,
        min_silence_len=0.2,
        breath_cutoff=100,
        target_dBFS=-16.0
    )