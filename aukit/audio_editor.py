# author: kdd
# date: 
"""
### audio_editor
语音编辑，切分音频，去除语音中的较长静音，去除语音首尾静音，设置采样率，设置通道数。
切分音频，去除静音，去除首尾静音输入输出都支持wav格式。
语音编辑功能基于pydub的方法，增加了数据格式支持。
"""
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from .audio_io import anything2bytesio, _sr
import numpy as np

_int16_max = 2 ** 15 - 1

set_channels = AudioSegment.set_channels
set_sample_rate = AudioSegment.set_frame_rate
set_set_sample_width = AudioSegment.set_sample_width


def audiosegment2wav(data: AudioSegment):
    wav = np.array(data.get_array_of_samples()) / _int16_max
    return wav


def wav2audiosegment(wav: np.ndarray, sr):
    tmp = anything2bytesio(wav, sr=sr)
    out = AudioSegment.from_wav(tmp)
    return out


def strip_silence_wave(wav: np.ndarray, sr=_sr, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    """
    去除语音前后静音。
    :param wav:
    :param sr:
    :param keep_silence_len:
    :param min_silence_len:
    :param silence_thresh:
    :param kwargs:
    :return:
    """
    data = wav2audiosegment(wav, sr=sr)
    out = strip_audio(data,
                      keep_silence_len=keep_silence_len,
                      min_silence_len=min_silence_len,
                      silence_thresh=silence_thresh,
                      **kwargs)
    out = audiosegment2wav(out)
    return out


def strip_audio(data: AudioSegment, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    nsils = detect_nonsilent(data, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if len(nsils) >= 1:
        return data[max(0, nsils[0][0] - keep_silence_len): min(len(data), nsils[-1][1] + keep_silence_len)]
    else:
        return AudioSegment.empty()


def strip_audio_os(inpath, outpath, **kwargs):
    try:
        data = AudioSegment.from_file(inpath, kwargs.get('format', 'wav'))
        out = strip_audio(data, **kwargs)
        out.export(outpath, kwargs.get('format', 'wav'))
    except Exception as e:
        print('Error path:', inpath)
        print('Error info:', e)


def split_silence_wave(wav, sr=_sr, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    """
    根据静音切分音频。
    :param wav:
    :param sr:
    :param keep_silence_len:
    :param min_silence_len:
    :param silence_thresh:
    :param kwargs:
    :return:
    """
    data = wav2audiosegment(wav, sr=sr)
    outs = split_audio(data,
                       keep_silence_len=keep_silence_len,
                       min_silence_len=min_silence_len,
                       silence_thresh=silence_thresh,
                       **kwargs)
    out_wavs = []
    for out in outs:
        wav = audiosegment2wav(out)
        out_wavs.append(wav)
    return out_wavs


def split_audio(data: AudioSegment, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    nsils = detect_nonsilent(data, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if len(nsils) >= 1:
        outs = []
        for ab in nsils:
            out = data[max(0, ab[0] - keep_silence_len): min(len(data), ab[1] + keep_silence_len)]
            outs.append(out)
    else:
        outs = [AudioSegment.empty()]
    return outs


def remove_silence_wave(wav, sr=_sr, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    """
    去除音频中的静音段。
    :param wav:
    :param sr:
    :param keep_silence_len:
    :param min_silence_len:
    :param silence_thresh:
    :param kwargs:
    :return:
    """
    data = wav2audiosegment(wav, sr=sr)
    out = remove_silence_audio(data,
                               keep_silence_len=keep_silence_len,
                               min_silence_len=min_silence_len,
                               silence_thresh=silence_thresh,
                               **kwargs)
    out = audiosegment2wav(out)
    return out


def remove_silence_audio(data: AudioSegment, keep_silence_len=20, min_silence_len=100, silence_thresh=-32, **kwargs):
    nsils = detect_nonsilent(data, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    out = AudioSegment.empty()
    sf = 0
    for i, ab in enumerate(nsils):
        si = max(ab[0] - keep_silence_len, sf)
        ei = ab[1] + keep_silence_len
        out = out + data[si: ei]
        sf = ei
    return out


if __name__ == "__main__":
    print(__file__)
