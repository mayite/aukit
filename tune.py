#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: KDD
# @time: 2018-11-10
"""
## 调音
"""
from pydub import AudioSegment
from scipy.io import wavfile
from pyaudio import PyAudio
import os
import numpy as np
import io
import wave


def tune_os(inpath=r"", **kwargs):
    io_in = io.BytesIO(open(inpath, "rb").read())
    io_out = tune_pitch(io_in=io_in, rate=kwargs.get("pitchrate", 1))
    io_out = tune_speed(io_out, rate=kwargs.get("speedrate", 1))
    # outpath = os.path.splitext(inpath)[0] + "_speed.wav"
    outpath = r"data/temp.wav"
    with open(outpath, "wb") as fout:
        fout.write(io_out.getvalue())
    play_audio_io(outpath)


def tune_speed(io_in=None, rate=1.):
    """
    变语速
    rate = win / (bar - cro)
    :param io_in:
    :param rate:
    :return:
    """

    song = AudioSegment.from_wav(io_in)
    n_song = len(song)
    win = 50
    bar = 100
    cro = int(bar - win / rate)

    segs = []
    for i in range(0, n_song - bar, win):
        segs.append(song[i: i + bar])

    out_song = segs[0]
    for seg in segs[1:]:
        out_song = out_song.append(seg, cro)

    io_out = io.BytesIO()
    out_song.export(io_out, format="wav")
    return io_out


def tune_pitch(io_in=None, rate=1.):
    """
    变音调
    :param io_in:
    :param rate:
    :return:
    """
    frate, wavdata = wavfile.read(io_in)

    cho_ids = [int(w) for w in np.arange(0, len(wavdata), rate)]
    out_wavdata = wavdata[cho_ids]

    io_out = io.BytesIO()
    wavfile.write(io_out, frate, out_wavdata)
    io_out = tune_speed(io_in=io_out, rate=1 / rate)

    return io_out


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


def play_tune(indir="", speedrate=1., pitchrate=1.):
    fname_list = [w for w in os.listdir(indir)]
    fname_list_new = sorted(fname_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    for fname in fname_list_new:
        inpath = os.path.join(indir, fname)
        tune_os(inpath=inpath, speedrate=speedrate, pitchrate=pitchrate)


if __name__ == "__main__":
    print(__file__)
    indir = r"D:\data\xinqing_wavs"
    play_tune(indir, speedrate=1, pitchrate=1)
    # inpath = r"D:\git\tts\data\beautiful_duo.wav"
    # for rate in np.arange(0.7, 1.7, 0.1):
    #     tune_os(inpath=inpath, speedrate=1., pitchrate=1)
