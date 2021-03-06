#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
### audio_io
语音IO，语音保存、读取，语音格式转换，支持【.】操作符的字典。
"""
from scipy.io import wavfile
from pathlib import Path
import numpy as np
import librosa
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.splitext(os.path.basename(__name__))[0])
_sr = 16000


class Dict2Obj(dict):
    """支持【.】操作符的dict。"""
    def __init__(self, *args, **kwargs):
        """
        字典的key是合法的变量名字符串才能支持【.】操作符。
        :param args:
        :param kwargs:
        """
        super(Dict2Obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


def load_wav(path, sr=None, with_sr=False):
    """
    导入语音信号。
    :param path: 文件路径。
    :param sr: 采样率，None: 自动识别采样率。
    :param with_sr: 是否返回采样率。
    :return: np.ndarray
    """
    if sr is not None:
        sr = sr or _sr
        return load_wav_librosa(path, sr=sr, with_sr=with_sr)
    else:
        return load_wav_wavfile(path, sr=sr, with_sr=with_sr)


def save_wav(wav, path, sr=None):
    save_wav_wavfile(wav, path=path, sr=sr)


def load_wav_librosa(path, sr=_sr, with_sr=False):
    wav, sr = librosa.core.load(path, sr=sr)
    return (wav, sr) if with_sr else wav


def load_wav_wavfile(path, sr=None, with_sr=False):
    sr, wav = wavfile.read(path)
    wav = wav / np.max(np.abs(wav))
    return (wav, sr) if with_sr else wav


def save_wav_librosa(wav, path, sr=_sr):
    librosa.output.write_wav(path, wav, sr=sr)


def save_wav_wavfile(wav, path, sr=_sr, volume=1.):
    out = wav * 32767 * volume / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, out.astype(np.int16))


def anything2bytesio(src, sr=_sr, volume=1.):
    if isinstance(src, (str, Path)):
        src = load_wav(src, sr=sr)
    if isinstance(src, (list, np.ndarray, np.matrix)):
        out_io = io.BytesIO()
        save_wav_wavfile(src, out_io, sr=sr, volume=volume)
    elif isinstance(src, bytes):
        out_io = io.BytesIO(src)
    elif isinstance(src, io.BytesIO):
        out_io = src
    else:
        raise TypeError
    return out_io


def anything2wav(src, sr=_sr, volume=1.):
    if isinstance(src, (list, np.ndarray, np.matrix)):
        return np.array(src)
    else:
        bysio = anything2bytesio(src, sr=sr, volume=volume)
        return load_wav_wavfile(bysio, sr=sr)


def anything2bytes(src, sr=_sr, volume=1.):
    if isinstance(src, bytes):
        return src
    else:
        bysio = anything2bytesio(src, sr=sr, volume=volume)
        return bysio.getvalue()


if __name__ == "__main__":
    print(__file__)
    inpath = r"E:\data\temp\01.wav"
    bys = anything2bytesio(inpath, sr=16000)
    print(bys)
    wav = anything2wav(bys, sr=16000)
    print(wav)
