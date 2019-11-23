#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/11/23
"""
"""
from pyaudio import PyAudio
import os
import wave


def play_audio_io(inpath):
    """
    播放音频
    :param inpath:
    :return:
    """
    print(f"正在播放：{inpath}")
    wf = wave.open(inpath, 'rb')
    play_audio(wf)


def play_audio(data=None):
    chunk = 1024  # 2014kb
    wf = data
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取数据

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
