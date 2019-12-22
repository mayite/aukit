#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
语音保存、读取。
"""
from scipy.io import wavfile
import numpy as np
import librosa
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.splitext(os.path.basename(__name__))[0])


def load_wav(path, sr=None, with_sr=False):
    try:
        if sr is None:
            return load_wav_wavfile(path, with_sr=with_sr)
        else:
            return load_wav_librosa(path, sr=sr, with_sr=with_sr)
    except Exception:
        return load_wav_librosa(path, sr=sr, with_sr=with_sr)


def save_wav(wav, path, sr=None):
    save_wav_wavfile(wav, path=path, sr=sr)


def load_wav_librosa(path, sr, with_sr=False):
    wav = librosa.core.load(path, sr=sr)[0]
    return (wav, sr) if with_sr else wav


def load_wav_wavfile(path, sr=None, with_sr=False):
    sr, wav = wavfile.read(path)
    wav = wav / np.max(np.abs(wav))
    return (wav, sr) if with_sr else wav


def save_wav_librosa(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def save_wav_wavfile(wav, path, sr):
    out = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, out.astype(np.int16))


def anything2bytesio(src, sr=None):
    if type(src) in {list, np.array, np.ndarray, np.matrix, np.asarray}:
        out_io = io.BytesIO()
        save_wav(src, out_io, sr)
    elif type(src) in {bytes}:
        out_io = io.BytesIO(src)
    elif type(src) in {str}:
        out_io = io.BytesIO(open(src, "rb").read())
    elif type(src) in {io.BytesIO}:
        out_io = src
    else:
        raise TypeError
    return out_io


if __name__ == "__main__":
    print(__file__)
