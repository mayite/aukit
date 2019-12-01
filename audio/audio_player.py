#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/23
"""
播放音频信号。
"""
from pyaudio import PyAudio
from scipy.io import wavfile
import io
import wave
import numpy as np


def save_wav(wav, path, sr):
    out = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, out.astype(np.int16))


def anything2bytesio(src, sr=None):
    if type(src) in {list, np.array, np.ndarray, np.matrix}:
        out_io = io.BytesIO()
        save_wav(src, out_io, sr)
    elif type(src) in {bytes}:
        out_io = io.BytesIO(src)
    elif type(src) in {str}:
        out_io = io.BytesIO(open(src, "rb").read())
    else:
        raise TypeError
    return out_io


def play_audio(src=None, sr=16000):
    chunk = 1024  # 2014kb
    bytesio = anything2bytesio(src, sr=sr)
    wf = wave.open(bytesio, "rb")
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    while True:
        data = wf.readframes(chunk)
        if data == b"":
            break
        stream.write(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio
    print('播放结束！')


if __name__ == "__main__":
    print(__file__)
