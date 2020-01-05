#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/1/5
"""
### audio_cli
命令行，播放音频。
"""
import sys
from . import play_audio


def play_audio_cli():
    fpath = sys.argv[1]
    sr = sys.argv[2]
    play_audio(fpath, sr=sr)


if __name__ == "__main__":
    print(__file__)
