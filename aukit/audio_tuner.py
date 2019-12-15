#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: KDD
# @time: 2018-11-10
"""
调节语速，调节音调。
"""
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import io


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


if __name__ == "__main__":
    print(__file__)
