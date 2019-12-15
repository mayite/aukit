# author: kdd
# date: 
"""
编辑音频，去除静音，切分音频。
"""
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def strip_audio(data: AudioSegment, keep_silence_len=200, min_silence_len=200, silence_thresh=-32, **kwargs):
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


def split_audio(data: AudioSegment, keep_silence_len=200, min_silence_len=200, silence_thresh=-32, **kwargs):
    nsils = detect_nonsilent(data, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if len(nsils) >= 1:
        outs = []
        for ab in nsils:
            out = data[max(0, ab[0] - keep_silence_len): min(len(data), ab[1] + keep_silence_len)]
            outs.append(out)
    else:
        outs = [AudioSegment.empty()]
    return outs


def remove_silence_audio(data: AudioSegment, keep_silence_len=200, min_silence_len=200, silence_thresh=-32, **kwargs):
    nsils = detect_nonsilent(data, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    out = AudioSegment.empty()
    for ab in nsils:
        out = out + data[ab[0]: ab[1]]
    return out


if __name__ == "__main__":
    print(__file__)
