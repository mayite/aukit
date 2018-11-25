#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: KDD
# @time: 2018-11-10
"""
## 分段
"""
import os
import re
import json
import collections as clt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile


def data_split(data, alpha=10., win=1600):
    split_ids = []
    for i in range(len(data) - win):
        a = np.sum(np.abs(data[i: i + win]))
        if a < alpha:
            split_ids.append(i + win // 2)
    split_ids.append(len(data) - 1)
    out_data_list = []
    i_pre = 0
    for i in split_ids:
        if i - i_pre >= win:
            out_data_list.append(data[i_pre: i])
        i_pre = i
    return out_data_list


def run_data_split_os():
    indir = r"D:\data\xinqing_wavs"
    outdir = indir + "_split"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    info_dict = clt.defaultdict(dict)
    for fname in tqdm(os.listdir(indir)):
        inpath = os.path.join(indir, fname)
        frate, wavdata = wavfile.read(inpath)
        data_split_list = data_split(data=wavdata, win=frate // 10)
        dt = {"filepath": inpath, "frame_num": len(wavdata), "split_num": len(data_split_list), "split_data": {}}
        info_dict[fname].update(dt)
        print(dt)
        for i, data in enumerate(data_split_list, 1):
            out_fname = fname.replace(".wav", f"_{i}.wav")
            outpath = os.path.join(outdir, out_fname)
            wavfile.write(outpath, frate, data)
            info_dict[fname]["split_data"].update({out_fname: {"filepath": outpath, "frame_num": len(data)}})


def text_split(text="", patt=None):
    if patt is None:
        non_zi_set = {'，', '×', '"', '。', '、', '.', '：', '?', '’', '！', '）', '？', '·', ')',
                      ' ', '；', '\\', '“', '”', '（', ',', '—', '(', '…'}
        split_set = non_zi_set.difference({'"', '’', '.', '（', '）', ' ', '\\', '“', '”', ')', '('})
        split_str = r"[{}]+".format(r"".join(split_set))
        split_patt = re.compile(split_str)
    else:
        split_patt = patt
    segs = split_patt.split(text)
    segs = [w for w in segs if re.sub(r"\W", "", w)]
    return segs


def run_text_split_os():
    inpath = r"D:\git\tts\data\xinqing_idx2text.json"
    outpath = os.path.splitext(inpath)[0] + "_split.json"
    idx_text_dict = json.load(open(inpath, encoding="utf8"))
    idx_numtext_dict = clt.defaultdict(int)
    idx_texts_dict = {}
    for idx, text in idx_text_dict.items():
        segs = text_split(text)
        idx_texts_dict[idx] = segs
        # print(idx, len(segs))
        idx_numtext_dict[idx] = len(segs)
    json.dump(idx_texts_dict, open(outpath, "wt", encoding="utf8"), indent=4, ensure_ascii=False)


def split_compare_os():
    inpath = r"D:\git\tts\data\xinqing_idx2text.json"
    inpath = os.path.splitext(inpath)[0] + "_split.json"
    idx_texts_dict = json.load(open(inpath, encoding="utf8"))
    idx_numtext_dict = {k: len(v) for k, v in idx_texts_dict.items()}

    indir = r"D:\data\xinqing_split_wavs"
    idx_numwav_dict = clt.defaultdict(int)
    for fname in tqdm(os.listdir(indir)):
        inpath = os.path.join(indir, fname)
        segs = fname.split("_")
        idx_numwav_dict[segs[-2]] += 1

    flag_list = []
    for k, v in idx_numwav_dict.items():
        flag = v == idx_numtext_dict[k]
        flag_list.append(flag)
        if not flag:
            print(f"idx: {k}, text: {idx_numtext_dict[k]}, wav: {v}")
    print(np.average(flag_list))


if __name__ == "__main__":
    print(__file__)
