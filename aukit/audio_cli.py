#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/1/5
"""
### audio_cli
命令行，播放音频，去除背景噪声。
"""
import sys
from .audio_player import play_audio
from .audio_noise_remover import remove_noise_os


def play_audio_cli():
    fpath = sys.argv[1]
    sr = sys.argv[2]
    play_audio(fpath, sr=sr)


def remove_noise_cli():
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    sr = int(sys.argv[3])
    remove_noise_os(inpath=inpath, outpath=outpath, sr=sr)


if __name__ == "__main__":
    print(__file__)
