#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/22
"""
变音。
"""
import numpy as np
import librosa


def spread(D, size=(3, 3)):
    """传播，重复每个数据点。"""
    if isinstance(size, tuple):
        if size[0] > 1:
            D = np.repeat(D, size[0], axis=0)
        if size[1] > 1:
            D = np.repeat(D, size[1], axis=1)
        if size[0] * size[1] > 1:
            D = D / (size[0] * size[1])
    elif isinstance(size, int):
        D = np.repeat(D, size, axis=1)
    return D


def change_pitch(n, y, sr):
    """
    调音高。
    :param n:-20~20,float，0:原声
    :param y:
    :param sr:
    :return:
    """
    return librosa.effects.pitch_shift(y, sr, n_steps=n)


def change_speed(n, y, sr):
    """
    调语速。
    :param n:0~5,float，0:原声
    :param y:
    :param sr:
    :return:
    """
    return librosa.effects.time_stretch(y, n)


def change_sample(n, y, sr):
    """
    调采样率，语速和音高同时改变。
    :param n:0~5,float，1:原声
    :param y:
    :param sr:
    :return:
    """
    return librosa.resample(y, sr, int(sr * n))


def change_reback(n, y, sr):
    """
    回声。
    :param n:1~10,int，1:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = pool(D, size=(1, n))
    D = repeat(D, n)
    return librosa.istft(D)


def change_pitchspeed(n, y, sr):
    """
    音高和语速同时变化。
    :param n:0~10,float，1:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    n = int(D.shape[0] * n)
    if n <= D.shape[0]:
        D = drop(D, D.shape[0] - n, mode="r")
    else:
        D = rewardshape(D, (n, D.shape[1]))
    return librosa.istft(D)


def change_attention(n, y, sr):
    """
    突出高音或低音段。
    :param n:-100~100,int，0:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = roll(D, n)
    return librosa.istft(D)


def change_male(n, y, sr):
    """
    变男声。
    :param n:0~1025,int，0,1,1025:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = pool_step(D, n)
    return librosa.istft(D)


def change_stretch(n, y, sr):
    """
    成倍拉伸延长。
    :param n:1~10,int，1:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = spread(D, n)
    return librosa.istft(D)


def change_vague(n, y, sr):
    """
    模糊。
    :param n:1~10,int，1:原声
    :param y:
    :param sr:
    :return:
    """
    frequencies, D = librosa.ifgram(y, sr=sr)
    D = pool(D, (1, n))
    D = spread(D, (1, n))
    return librosa.istft(D)


class CheckStep(object):
    def __init__(self, step):
        self.step = step
        self.index = 0

    def __call__(self, *args):
        self.index += 1
        return self.index % self.step != 0


def drop(D, n, mode="l"):
    """丢弃,mode:left,right,side,center"""
    if n == 0:
        return D
    if mode == "l":
        return D[n:]
    elif mode == "r":
        return D[:-n]
    elif mode == "s":
        return D[n // 2:-(n // 2)]
    elif mode == "c":
        if n < len(D):
            return np.vstack((D[:n // 2], D[-(n // 2):]))
        else:
            return ()
    else:
        raise AssertionError


def repeat(D, n, axis=0):
    """重复"""
    return np.repeat(D, n, axis=axis)


def roll(D, n):
    """循环移动"""
    return np.roll(D, n, axis=0)


def rewardshape(D, shape):
    """填充"""
    x = shape[0] - D.shape[0]
    y = shape[1] - D.shape[1]
    if x > 0:
        bottomlist = np.zeros([x, D.shape[1]])
        D = np.r_[D, bottomlist]
    if y > 0:
        rightlist = np.zeros([D.shape[0], y])
        D = np.c_[D, rightlist]
    return D


def pool_step(D, step):
    """步长池化"""
    _shape = D.shape
    if step < 2:
        return D
    cs = CheckStep(step)
    return rewardshape(np.array(list(filter(cs, D))), _shape)


def pool(D, size=(3, 3), shapeed=False):
    """池化"""
    _shape = D.shape
    if isinstance(size, tuple):
        if size[1] > 1:
            D = _pool(D, size[1])
        if size[0] > 1:
            D = _pool(D.T, size[0]).T
    elif isinstance(size, int):
        D = _pool(D.T, size).T
    if shapeed:
        D = rewardshape(D, _shape)
    return D


def _pool(D, poolsize):
    """池化方法"""
    x = D.shape[1] // poolsize
    restsize = D.shape[1] % poolsize
    if restsize > 0:
        x += 1
        rightlist = np.zeros([D.shape[0], poolsize - restsize])
        D = np.c_[D, rightlist]
    D = D.reshape((-1, poolsize))
    D = D.sum(axis=1).reshape(-1, x)
    return D


if __name__ == '__main__':
    from aukit.audio_player import play_audio

    path = r"E:/data/temp/01.wav"
    y, sr = librosa.load(path)

    # y = change_vague(3, y, sr)
    # y = change_pitch(-6, y, sr)
    # y = change_speed(0.5, y, sr)
    # y = change_sample(0.8, y, sr)
    # y = change_reback(3, y, sr)
    # y = change_pitchspeed(2, y, sr)
    # y = change_attention(50, y, sr)
    # y = change_male(5, y, sr)
    # y = change_vague(6, y, sr)

    """童声"""
    y = change_pitch(6, y, sr)
    y = change_male(20, y, sr)

    play_audio(y, sr=sr)
