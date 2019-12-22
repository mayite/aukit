#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/23
"""
播放音频信号。
"""
from pyaudio import PyAudio
from scipy.io import wavfile
import sys
import io
import wave
import numpy as np
import sounddevice as sd
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.splitext(os.path.basename(__name__))[0])

from .audio_io import load_wav, anything2bytesio


def play_audio(src=None, sr=16000):
    chunk = 1024  # 2014kb
    bytesio = anything2bytesio(src, sr=sr)
    wf = wave.open(bytesio, "rb")
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    t0 = time.time()
    while True:
        data = wf.readframes(chunk)
        if data == b"":
            break
        stream.write(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio
    t = time.time() - t0
    logger.info("play audio done, playing {:.2f} seconds.".format(t))


def play_audio_cmd():
    fpath = sys.argv[1]
    sr = sys.argv[2]
    play_audio(fpath, sr)


def play_sound(src, sr=16000, **kwargs):
    if type(src) not in {np.asarray, np.array, np.ndarray, list}:
        bytesio = anything2bytesio(src, sr=sr)
        data = load_wav(bytesio, sr=sr)
    else:
        data = np.array(src)
    t0 = time.time()
    sd.play(data, sr, **kwargs)
    sd.wait()
    t = time.time() - t0
    logger.info("play sound done, playing {:.2f} seconds.".format(t))


if __name__ == "__main__":
    print(__file__)
