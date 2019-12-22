#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/30
"""
语音频谱，语音转线性频谱，语音转梅尔频谱。
"""
from scipy.signal import lfilter
import math
import numpy as np
import librosa


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
    # Audio
    mel_basis=None,
    inv_mel_basis=None,
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality

    # Mel spectrogram
    n_fft=2048,  # 800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    preemphasize=False,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.

    # Griffin Lim
    power=1.5,
))


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def save_wav(x, path, sr):
    librosa.output.write_wav(path, x.astype(np.float32), sr=sr)


def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
    return np.clip(x * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int16)


def linear_to_mel(spectrogram, hparams=None):
    hparams = hparams or default_hparams
    if hparams.mel_basis is None:
        mel_basis = build_mel_basis(hparams)
    else:
        mel_basis = hparams.mel_basis
    return np.dot(mel_basis, spectrogram)


def mel_to_linear(mel_spectrogram, hparams=None):
    hparams = hparams or default_hparams
    if hparams.inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(build_mel_basis(hparams))
    else:
        _inv_mel_basis = hparams.inv_mel_basis
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def build_mel_basis(hparams=None):
    hparams = hparams or default_hparams
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels, fmin=hparams.fmin)


def normalize(S, hparams=None):
    hparams = hparams or default_hparams
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def denormalize(S, hparams=None):
    hparams = hparams or default_hparams
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def linear_spectrogram(y, hparams=None):
    hparams = hparams or default_hparams
    D = stft(pre_emphasis(y, hparams), hparams)
    S = amp_to_db(np.abs(D)) - hparams.ref_level_db
    return normalize(S, hparams)


def mel_spectrogram(y, hparams=None):
    hparams = hparams or default_hparams
    D = stft(pre_emphasis(y, hparams), hparams)
    S = amp_to_db(linear_to_mel(np.abs(D), hparams))
    return normalize(S, hparams)


def mel_spectrogram_feature(wav, hparams=None):
    hparams = hparams or default_hparams
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        hparams.sample_rate,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_size,
        n_mels=hparams.num_mels
    )
    return amp_to_db(frames.astype(np.float32))


def stft(y, hparams=None):
    hparams = hparams or default_hparams
    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size)


def pre_emphasis(x, hparams=None):
    hparams = hparams or default_hparams
    if hparams.preemphasize:
        return lfilter([1, -hparams.preemphasis], [1], x)
    else:
        return x


def de_emphasis(x, hparams=None):
    hparams = hparams or default_hparams
    if hparams.preemphasize:
        return lfilter([1], [1, -hparams.preemphasis], x)
    else:
        return x


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


if __name__ == "__main__":
    print(__file__)
