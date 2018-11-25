#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: KDD
# @file: run.py
# @time: 2018-11-10
"""
APPID
5be6dcdd
APIKey
a65d8fda71993073b0c529033085df2d
IP白名单
112.96.173.60
"""
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import pyworld as pw
from scipy.io import wavfile
import os
import re
import json
import collections as clt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")

TESTWAVPATH = os.path.join(DATADIR, "beautiful_duo.wav")


def test_world_syn():
    # 读取文件
    WAV_FILE = TESTWAVPATH

    # 提取语音特征
    x, fs = sf.read(WAV_FILE)

    # f0 : ndarray
    #     F0 contour. 基频等高线
    # sp : ndarray
    #     Spectral envelope. 频谱包络
    # ap : ndarray
    #     Aperiodicity. 非周期性
    f0, sp, ap = pw.wav2world(x, fs)  # use default options

    # 分布提取参数
    # 使用DIO算法计算音频的基频F0

    _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0, channels_in_octave=2, frame_period=pw.default_frame_period)

    # 使用CheapTrick算法计算音频的频谱包络

    _sp = pw.cheaptrick(x, _f0, t, fs)

    # 计算aperiodic参数

    _ap = pw.d4c(x, _f0, t, fs)

    # 基于以上参数合成音频

    _y = pw.synthesize(_f0, _sp, _ap, fs, pw.default_frame_period)

    # 写入音频文件

    sf.write('test/y_without_f0_refinement.wav', _y, fs)

    plt.plot(f0)
    plt.show()
    plt.imshow(np.log(sp), cmap='gray')
    plt.show()
    plt.imshow(ap, cmap='gray')

    # 合成原始语音
    synthesized = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

    # 1.输出原始语音
    sf.write('test/synthesized.wav', synthesized, fs)

    # 2.变高频-更类似女性
    high_freq = pw.synthesize(f0 * 2.0, sp, ap, fs)  # 周波数を2倍にすると1オクターブ上がる

    sf.write('test/high_freq.wav', high_freq, fs)

    # 3.直接修改基频，变为机器人发声
    robot_like_f0 = np.ones_like(f0) * 100  # 100は適当な数字
    robot_like = pw.synthesize(robot_like_f0, sp, ap, fs)

    sf.write('test/robot_like.wav', robot_like, fs)

    # 4.提高基频，同时频谱包络后移？更温柔的女性？
    female_like_sp = np.zeros_like(sp)
    for f in range(female_like_sp.shape[1]):
        female_like_sp[:, f] = sp[:, int(f / 1.2)]
    female_like = pw.synthesize(f0 * 2, female_like_sp, ap, fs)

    sf.write('test/female_like.wav', female_like, fs)

    # 5.转换基频（不能直接转换）
    x2, fs2 = sf.read(TESTWAVPATH)
    f02, sp2, ap2 = pw.wav2world(x2, fs2)
    f02 = f02[:len(f0)]
    print(len(f0), len(f02))
    other_like = pw.synthesize(f02, sp, ap, fs)

    sf.write('test/other_like.wav', other_like, fs)


def test_world():
    # 获取音频的采样点数值以及采样率
    x, fs = sf.read(TESTWAVPATH)

    # 使用DIO算法计算音频的基频F0

    _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0, channels_in_octave=2)
    #    , frame_period=args.frame_period,                  speed=args.speed)

    # 使用CheapTrick算法计算音频的频谱包络

    _sp = pw.cheaptrick(x, _f0, t, fs)

    # 计算aperiodic参数

    _ap = pw.d4c(x, _f0, t, fs)

    # 基于以上参数合成音频

    _y = pw.synthesize(_f0, _sp, _ap, fs)
    # , args.frame_period)

    # 写入音频文件
    outpath = os.path.splitext(TESTWAVPATH)[0] + "_w.wav"
    sf.write(outpath, _y, fs)


def test_spec():
    from tacotron.util.audio import wave2mel, spec2wave, wave2spec, load_wav, save_wav
    wavdata = load_wav(TESTWAVPATH)
    spec = wave2spec(wavdata)
    print(spec.shape)
    wavdata = spec2wave(spec)
    outpath = TESTWAVPATH[:-4] + "_spec.wav"
    save_wav(wavdata, outpath)


def test_rename_os():
    indir = r"D:\data\xunfei_wavs"
    outdir = r"D:\data\interview_questions_wavs"
    for fname in os.listdir(indir):
        fpath = os.path.join(indir, fname)
        segs = fname.split("_")
        out_fname = "_".join([segs[0], "interview_questions", segs[-1]])
        outpath = os.path.join(outdir, out_fname)
        os.rename(fpath, outpath)


def test_pydub_os():
    from pydub import AudioSegment as auseg
    inpath = r"D:\git\tts\data\beautiful_duo.wav"
    song = auseg.from_wav(inpath)
    outpath = os.path.splitext(inpath)[0] + "_dub.wav"
    song.export(outpath, format="wav")
    from tune import play_audio
    play_audio(outpath)


if __name__ == "__main__":
    print(__file__)
    # test_world_syn()
    # test_world()
    # test_spec()
    # import wave
    #
    # wavdata = wave.open(TESTWAVPATH)
    # print(wavdata)
    # test_rename_os()
