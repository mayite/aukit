#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: KDD
# @file: xunfei.py
# @time: 2018-11-10
"""
APPID
5be6dcdd
APIKey
a65d8fda71993073b0c529033085df2d
IP白名单
112.96.173.60
"""

import os
import base64
import json
import time
import hashlib
import urllib.request
import urllib.parse
from tqdm import tqdm


# API请求地址、API KEY、APP ID等参数，提前填好备用
api_url = "http://api.xfyun.cn/v1/service/v1/tts"
API_KEY = "a65d8fda71993073b0c529033085df2d"#""替换成你的APIKey"
APP_ID = "5be6dcdd"#"替换成你的APPID"
OUTPUT_FILE = "/home/duoyi/PycharmProjects/tts/data/xunfei_wavs/beautiful_duo.wav"    # 输出音频的保存路径，请根据自己的情况替换
TEXT = "我家朵朵是世界上最漂亮的朵朵，世界上最漂亮的朵朵就是我家朵朵。"
OUTPUTDIR = "/home/duoyi/PycharmProjects/tts/data/xunfei_wavs"
# 构造输出音频配置参数
Param = {
    "auf": "audio/L16;rate=16000",    #音频采样率
    "aue": "raw",    #音频编码，raw(生成wav)或lame(生成mp3)
    "voice_name": "xiaoyan",
    "speed": "50",    #语速[0,100]
    "volume": "77",    #音量[0,100]
    "pitch": "50",    #音高[0,100]
    "engine_type": "aisound"    #引擎类型。aisound（普通效果），intp65（中文），intp65_en（英文）
}
# 配置参数编码为base64字符串，过程：字典→明文字符串→utf8编码→base64(bytes)→base64字符串
Param_str = json.dumps(Param)    #得到明文字符串
Param_utf8 = Param_str.encode('utf8')    #得到utf8编码(bytes类型)
Param_b64 = base64.b64encode(Param_utf8)    #得到base64编码(bytes类型)
Param_b64str = Param_b64.decode('utf8')    #得到base64字符串

# 构造HTTP请求的头部
time_now = str(int(time.time()))
checksum = (API_KEY + time_now + Param_b64str).encode('utf8')
checksum_md5 = hashlib.md5(checksum).hexdigest()
header = {
    "X-Appid": APP_ID,
    "X-CurTime": time_now,
    "X-Param": Param_b64str,
    "X-CheckSum": checksum_md5
}


def test():
    # 构造HTTP请求Body
    body = {
        "text": TEXT
    }
    body_urlencode = urllib.parse.urlencode(body)
    body_utf8 = body_urlencode.encode('utf8')

    # 发送HTTP POST请求
    req = urllib.request.Request(api_url, data=body_utf8, headers=header)
    response = urllib.request.urlopen(req)

    # 读取结果
    response_head = response.headers['Content-Type']
    if(response_head == "audio/mpeg"):
        out_file = open(OUTPUT_FILE, 'wb')
        data = response.read() # a 'bytes' object
        out_file.write(data)
        out_file.close()
        print('输出文件: ' + OUTPUT_FILE)
    else:
        print(response.read().decode('utf8'))


def getwave(text=""):
    # 构造HTTP请求Body
    body = {
        "text": text
    }
    body_urlencode = urllib.parse.urlencode(body)
    body_utf8 = body_urlencode.encode('utf8')

    # 发送HTTP POST请求
    req = urllib.request.Request(api_url, data=body_utf8, headers=header)
    response = urllib.request.urlopen(req)

    # 读取结果
    response_head = response.headers['Content-Type']
    if(response_head == "audio/mpeg"):
        data = response.read() # a 'bytes' object
        return data
    else:
        print(response.read().decode('utf8'))
        return None


def savewave(data=None, path=""):
    if data is None:
        return None
    with open(path, "wb") as fout:
        fout.write(data)
    return data


def run(idx_text_dict=None, idx_path_dict=None):
    idx_lst = sorted(idx_text_dict)
    for idx in tqdm(idx_lst):
        if os.path.exists(idx_path_dict[idx]):
            continue
        data = getwave(idx_text_dict[idx])
        savewave(data, idx_path_dict[idx])

def getdicts(inpath=""):
    idx_text_dict = {}
    idx_path_dict = {}
    with open(inpath) as fin:
        for idx, text in enumerate(fin):
            idx_text_dict[idx] = text.strip()
            idx_path_dict[idx] = os.path.join(OUTPUTDIR, "xunfei_{}.wav".format(idx))
    return idx_text_dict, idx_path_dict



if __name__ == "__main__":
    print(__file__)

    # test()

    inpath = "/home/duoyi/PycharmProjects/tts/data/interview_questions.txt"
    itdt, ipdt = getdicts(inpath)

    itpath = os.path.splitext(inpath)[0] + "_idx2text.json"
    json.dump(itdt, open(itpath, "wt"), indent=4, ensure_ascii=False)
    ippath = os.path.splitext(inpath)[0] + "_idx2path.json"
    json.dump(ipdt, open(ippath, "wt"), indent=4, ensure_ascii=False)

    run(itdt, ipdt)
