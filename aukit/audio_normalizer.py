#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/30
"""
语音正则化，去除语音中音量低的部分，标准化音量。
"""
from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct


class Dict2Obj(dict):
    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


# Default hyperparameters
default_hparams = Dict2Obj(dict(
    int16_max=(2 ** 15) - 1,
    ## Mel-filterbank
    mel_window_length=25,  # In milliseconds
    mel_window_step=10,  # In milliseconds
    mel_n_channels=40,

    ## Audio
    sample_rate=16000,  # sampling_rate
    # Number of spectrogram frames in a partial utterance
    partials_n_frames=160,  # 1600 ms
    # Number of spectrogram frames at inference
    inference_n_frames=80,  # 800 ms

    ## Voice Activation Detection
    # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length=30,  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_moving_average_width=8,
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length=6,

    ## Audio volume normalization
    audio_norm_target_dBFS=-30,
))


def hparams_debug_string(hparams):
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None, hparams=None):
    hparams = hparams or default_hparams
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != hparams.sample_rate:
        wav = librosa.resample(wav, source_sr, hparams.sample_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, hparams.audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


def wav_to_mel_spectrogram(wav, hparams=None):
    hparams = hparams or default_hparams
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        hparams.sample_rate,
        n_fft=int(hparams.sample_rate * hparams.mel_window_length / 1000),
        hop_length=int(hparams.sample_rate * hparams.mel_window_step / 1000),
        n_mels=hparams.mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav, hparams=None):
    hparams = hparams or default_hparams
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (hparams.vad_window_length * hparams.sample_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * hparams.int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=hparams.sample_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False, hparams=None):
    hparams = hparams or default_hparams
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * hparams.int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / hparams.int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


if __name__ == "__main__":
    print(__file__)
