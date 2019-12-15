#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
world声码器。
"""
import pyworld as pw


def world_spectrogram_default(x, sr):
    """默认参数的world声码器语音转为特征频谱。"""
    # f0 : ndarray
    #     F0 contour. 基频等高线
    # sp : ndarray
    #     Spectral envelope. 频谱包络
    # ap : ndarray
    #     Aperiodicity. 非周期性
    f0, sp, ap = pw.wav2world(x, sr)  # use default options
    return f0, sp, ap


def inv_world_spectrogram_default(f0, sp, ap, sr):
    """默认参数的world声码器特征频谱转为语音。"""
    y = pw.synthesize(f0, sp, ap, sr)
    return y


def world_spectrogram(x, sr, dim_num=32, **kwargs):
    """world声码器语音转为频谱。"""
    # 分布提取参数
    # 使用DIO算法计算音频的基频F0
    f0, t = pw.dio(x, sr, **kwargs)
    # f0, t = pw.harvest(x, sr)

    # 使用CheapTrick算法计算音频的频谱包络
    sp = pw.cheaptrick(x, f0, t, sr, **kwargs)
    # SP降维
    sp_enc = pw.code_spectral_envelope(sp, sr, number_of_dimensions=dim_num)

    # 计算aperiodic参数
    ap = pw.d4c(x, f0, t, sr, **kwargs)
    # AP降维
    ap_enc = pw.code_aperiodicity(ap, sr)
    return f0, sp_enc, ap_enc


def inv_world_spectrogram(f0, sp, ap, sr, frame_period=5, fft_size=1024):
    """world声码器频谱转为语音。"""
    sp_dec = pw.decode_spectral_envelope(sp, sr, fft_size=fft_size)
    ap_dec = pw.decode_aperiodicity(ap, sr, fft_size=fft_size)
    y = pw.synthesize(f0, sp_dec, ap_dec, sr, frame_period=frame_period)
    return y


if __name__ == "__main__":
    print(__file__)
