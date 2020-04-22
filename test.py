#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
import time
import numpy as np
from aukit import _sr

# 普
_wav = np.array([
    0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, -0.01, -0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04,
    0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.11, 0.11, 0.11, 0.1, 0.09, 0.08, 0.06, 0.04, 0.02, 0.0, -0.03, -0.06, -0.09,
    -0.11, -0.14, -0.18, -0.2, -0.23, -0.26, -0.28, -0.3, -0.32, -0.33, -0.34, -0.34, -0.33, -0.32, -0.3, -0.28, -0.24,
    -0.21, -0.18, -0.14, -0.09, -0.04, 0.0, 0.03, 0.07, 0.1, 0.13, 0.15, 0.17, 0.19, 0.2, 0.21, 0.21, 0.2, 0.19, 0.18,
    0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03,
    0.02, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.09,
    0.1, 0.11, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.08, -0.1,
    -0.12, -0.15, -0.17, -0.2, -0.22, -0.24, -0.27, -0.29, -0.31, -0.32, -0.33, -0.34, -0.34, -0.33, -0.32, -0.3, -0.27,
    -0.24, -0.2, -0.16, -0.12, -0.07, -0.02, 0.01, 0.05, 0.09, 0.13, 0.15, 0.18, 0.2, 0.21, 0.22, 0.22, 0.22, 0.22,
    0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.13, 0.12, 0.11, 0.1, 0.1, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03,
    0.02, 0.01, 0.0, -0.01, -0.02, -0.02, -0.03, -0.04, -0.05, -0.05, -0.04, -0.04, -0.03, -0.02, -0.01, 0.0, 0.02,
    0.03, 0.05, 0.08, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.16, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.04, 0.02,
    0.0, -0.02, -0.04, -0.07, -0.09, -0.11, -0.13, -0.15, -0.18, -0.2, -0.22, -0.25, -0.28, -0.3, -0.32, -0.34, -0.35,
    -0.36, -0.36, -0.35, -0.34, -0.33, -0.31, -0.27, -0.24, -0.19, -0.15, -0.1, -0.04, 0.0, 0.04, 0.09, 0.13, 0.16,
    0.19, 0.21, 0.23, 0.25, 0.25, 0.25, 0.25, 0.25, 0.23, 0.22, 0.2, 0.19, 0.17, 0.15, 0.13, 0.12, 0.11, 0.1, 0.08,
    0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, -0.01, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.08, -0.07, -0.07,
    -0.06, -0.05, -0.03, -0.01, 0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.17, 0.18, 0.19, 0.2, 0.2, 0.19, 0.18, 0.17,
    0.16, 0.14, 0.11, 0.09, 0.06, 0.04, 0.02, 0.0, -0.02, -0.04, -0.06, -0.08, -0.1, -0.12, -0.13, -0.15, -0.18, -0.21,
    -0.22, -0.24, -0.27, -0.31, -0.33, -0.34, -0.36, -0.38, -0.39, -0.39, -0.38, -0.37, -0.35, -0.31, -0.27, -0.22,
    -0.17, -0.12, -0.06, 0.0, 0.04, 0.09, 0.14, 0.19, 0.23, 0.25, 0.27, 0.29, 0.3, 0.29, 0.28, 0.28, 0.26, 0.24, 0.22,
    0.2, 0.18, 0.15, 0.13, 0.11, 0.1, 0.08, 0.07, 0.05, 0.04, 0.04, 0.03, 0.02, 0.0, 0.0, 0.0, -0.02, -0.03, -0.04,
    -0.05, -0.07, -0.08, -0.08, -0.09, -0.09, -0.08, -0.07, -0.06, -0.05, -0.03, 0.0, 0.02, 0.05, 0.07, 0.1, 0.13, 0.16,
    0.18, 0.2, 0.22, 0.23, 0.23, 0.22, 0.21, 0.21, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.03, 0.01, -0.01, -0.03, -0.05,
    -0.06, -0.08, -0.09, -0.11, -0.13, -0.14, -0.16, -0.17, -0.21, -0.22, -0.25, -0.28, -0.31, -0.34, -0.36, -0.38,
    -0.4, -0.42, -0.42, -0.4, -0.38, -0.36, -0.34, -0.29, -0.23, -0.18, -0.12, -0.06, 0.0, 0.06, 0.11, 0.16, 0.21, 0.25,
    0.28, 0.3, 0.32, 0.33, 0.33, 0.32, 0.3, 0.28, 0.26, 0.23, 0.2, 0.18, 0.15, 0.13, 0.1, 0.08, 0.07, 0.06, 0.04, 0.03,
    0.03, 0.02, 0.01, 0.0, 0.0, -0.01, -0.02, -0.03, -0.05, -0.06, -0.07, -0.09, -0.1, -0.11, -0.11, -0.12, -0.11, -0.1,
    -0.08, -0.06, -0.05, -0.02, 0.0, 0.05, 0.07, 0.1, 0.14, 0.18, 0.21, 0.23, 0.24, 0.26, 0.27, 0.26, 0.24, 0.23, 0.21,
    0.19, 0.16, 0.13, 0.1, 0.07, 0.04, 0.01, 0.0, -0.02, -0.04, -0.06, -0.08, -0.08, -0.09, -0.11, -0.12, -0.13, -0.14,
    -0.17, -0.19, -0.21, -0.23, -0.25, -0.3, -0.34, -0.35, -0.37, -0.39, -0.41, -0.42, -0.41, -0.39, -0.39, -0.36,
    -0.31, -0.25, -0.19, -0.14, -0.08, 0.0, 0.06, 0.11, 0.16, 0.22, 0.27, 0.3, 0.32, 0.34, 0.35, 0.34, 0.33, 0.31, 0.3,
    0.27, 0.24, 0.2, 0.18, 0.16, 0.13, 0.09, 0.07, 0.06, 0.05, 0.03, 0.01, 0.01, 0.01, 0.0, -0.01, -0.02, -0.02, -0.03,
    -0.04, -0.06, -0.07, -0.08, -0.1, -0.12, -0.13, -0.13, -0.14, -0.14, -0.14, -0.12, -0.1, -0.08, -0.05, -0.02, 0.01,
    0.05, 0.09, 0.13, 0.18, 0.21, 0.24, 0.27, 0.29, 0.3, 0.31, 0.31, 0.29, 0.27, 0.24, 0.21, 0.17, 0.14, 0.1, 0.07,
    0.04, 0.0, -0.01, -0.03, -0.05, -0.07, -0.08, -0.09, -0.1, -0.11, -0.12, -0.12, -0.13, -0.15, -0.17, -0.19, -0.21,
    -0.24, -0.27, -0.31, -0.34, -0.36, -0.39, -0.42, -0.44, -0.44, -0.43, -0.41, -0.39, -0.36, -0.3, -0.24, -0.18,
    -0.13, -0.06, 0.02, 0.09, 0.14, 0.19, 0.25, 0.3, 0.33, 0.35, 0.36, 0.36, 0.35, 0.33, 0.32, 0.29, 0.25, 0.21, 0.19,
    0.16, 0.13, 0.09, 0.06, 0.04, 0.03, 0.01, 0.0, -0.01, -0.01, -0.01, -0.01, -0.02, -0.03, -0.03, -0.03, -0.04, -0.06,
    -0.07, -0.08, -0.09, -0.11, -0.12, -0.13, -0.14, -0.14, -0.14, -0.12, -0.1, -0.09, -0.07, -0.02, 0.02, 0.05, 0.08,
    0.13, 0.18, 0.23, 0.26, 0.28, 0.31, 0.34, 0.35, 0.33, 0.32, 0.3, 0.28, 0.24, 0.2, 0.17, 0.13, 0.08, 0.04, 0.01,
    -0.01, -0.04, -0.06, -0.07, -0.08, -0.08, -0.1, -0.11, -0.11, -0.1, -0.11, -0.13, -0.14, -0.15, -0.17, -0.21, -0.25,
    -0.28, -0.31, -0.34, -0.37, -0.4, -0.42, -0.44, -0.44, -0.43, -0.41, -0.39, -0.36, -0.3, -0.23, -0.17, -0.11, -0.04,
    0.03, 0.11, 0.17, 0.22, 0.27, 0.31, 0.34, 0.36, 0.37, 0.36, 0.35, 0.33, 0.31, 0.28, 0.24, 0.2, 0.17, 0.14, 0.1,
    0.07, 0.04, 0.02, 0.01, 0.0, 0.0, -0.02, -0.02, -0.01, -0.01, -0.02, -0.03, -0.03, -0.03, -0.04, -0.06, -0.07,
    -0.09, -0.1, -0.12
])

_wav_bytes = (
    b'RIFFd\x06\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00'
    b'\x00}\x00\x00\x02\x00\x10\x00data@\x06\x00\x00\x8b\x0e\xa2\x0b\xba\x08\xd1\x05\xe8\x02\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x18\xfd\x18\xfd\x18\xfd\x18\xfd\x00\x00\x00\x00\x00\x00\xe8\x02\xd1\x05\xba\x08\xa2\x0b'
    b't\x11\\\x14E\x17.\x1a\x17\x1d\xff\x1f\xff\x1f\xff\x1f\xff\x1f\x17\x1d.\x1aE\x17t\x11\xa2\x0b'
    b'\xd1\x05\x00\x00F\xf7\x8c\xee\xd2\xe5\x01\xe0G\xd7\xa4\xcb\xd2\xc5\x18\xbd^\xb4\x8d\xae\xbb\xa8\xea\xa2'
    b'\x01\xa0\x19\x9d\x19\x9d\x01\xa0\xea\xa2\xbb\xa8\x8d\xae0\xba\xea\xc2\xa4\xcbG\xd7\xd2\xe5^\xf4\x00\x00'
    b'\xba\x08\\\x14\x17\x1d\xd1%\xa2+s1E7.:\x16=\x16=.:E7\\4s1'
    b'\x8b.\xa2+\xb9(\xd1%\xe8"\xff\x1f\x17\x1d.\x1aE\x17E\x17\\\x14\\\x14t\x11\x8b\x0e'
    b'\x8b\x0e\xa2\x0b\xba\x08\xba\x08\xd1\x05\xe8\x02\xe8\x02\x00\x00\x00\x00\x00\x00\x00\x00\x18\xfd\x18\xfd\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\xe8\x02\xd1\x05\xa2\x0b\x8b\x0et\x11\\\x14.\x1a\x17\x1d\xff\x1f\xe8"\xe8"'
    b'\xe8"\xe8"\xff\x1f\xff\x1f\x17\x1d.\x1a\\\x14\x8b\x0e\xba\x08\xe8\x02\x18\xfdF\xf7u\xf1\xbb\xe8'
    b'\xe9\xe2\x18\xdd^\xd4\x8d\xce\xd2\xc5\x01\xc00\xbau\xb1\xa4\xab\xd3\xa5\xea\xa2\x01\xa0\x19\x9d\x19\x9d'
    b'\x01\xa0\xea\xa2\xbb\xa8u\xb10\xba\xd2\xc5u\xd1\x18\xdd\xa4\xeb/\xfa\xe8\x02\x8b\x0e.\x1a\xd1%'
    b'\xa2+\\4.:\x16=\xff?\xff?\xff?\xff?\x16=E7\\4s1\x8b.\xa2+'
    b'\xd1%\xe8"\xff\x1f\x17\x1d\x17\x1d.\x1aE\x17E\x17\\\x14t\x11\x8b\x0e\xa2\x0b\xba\x08\xd1\x05'
    b'\xe8\x02\x00\x00\x18\xfd/\xfa/\xfaF\xf7^\xf4u\xf1u\xf1^\xf4^\xf4F\xf7/\xfa\x18\xfd'
    b'\x00\x00\xd1\x05\xba\x08\x8b\x0eE\x17.\x1a\xff\x1f\xd1%\xb9(\xa2+\x8b.\x8b.\x8b.\xa2+'
    b'\xb9(\xd1%\xff\x1f.\x1a\\\x14\xa2\x0b\xd1\x05\x00\x00/\xfa^\xf4\xa4\xeb\xd2\xe5\x01\xe0/\xda'
    b'^\xd4\xa4\xcb\xd2\xc5\x01\xc0G\xb7\x8d\xae\xbb\xa8\xea\xa2\x19\x9d0\x9aG\x97G\x970\x9a\x19\x9d'
    b'\x01\xa0\xd3\xa5u\xb10\xba\xbb\xc8^\xd4\xe9\xe2^\xf4\x00\x00\xa2\x0b.\x1a\xd1%\x8b.E7'
    b'\x16=\xe8B\xb9H\xb9H\xb9H\xb9H\xb9H\xe8B\xff?.:E7s1\xa2+\xd1%'
    b'\xe8"\xff\x1f\x17\x1dE\x17\\\x14t\x11\x8b\x0e\xa2\x0b\xba\x08\xd1\x05\xe8\x02\x00\x00\x18\xfdF\xf7'
    b'^\xf4u\xf1\x8c\xee\xa4\xeb\xbb\xe8\xbb\xe8\xa4\xeb\xa4\xeb\x8c\xeeu\xf1F\xf7\x18\xfd\x00\x00\xd1\x05'
    b'\x8b\x0eE\x17\x17\x1d\xe8"\xa2+s1\\4E7.:.:E7\\4s1\x8b.'
    b'\xb9(\xff\x1f.\x1at\x11\xa2\x0b\xd1\x05\x00\x00/\xfa^\xf4\x8c\xee\xbb\xe8\xe9\xe2\x18\xdd/\xda'
    b'^\xd4\xa4\xcb\xea\xc2\x01\xc00\xbau\xb1\xd3\xa5\x01\xa0\x19\x9dG\x97v\x91\x8d\x8e\x8d\x8ev\x91'
    b'^\x940\x9a\xd3\xa5u\xb1\x01\xc0\x8d\xce\x18\xdd\x8c\xee\x00\x00\xa2\x0b.\x1a\xb9(E7\xe8B'
    b'\xb9H\x8bN\\TEW\\TsQsQ\xa2K\xd0E\xff?.:\\4\xa2+\xd1%'
    b'\xff\x1f\x17\x1dE\x17\\\x14\x8b\x0e\xa2\x0b\xa2\x0b\xba\x08\xd1\x05\x00\x00\x00\x00\x00\x00/\xfaF\xf7'
    b'^\xf4u\xf1\xa4\xeb\xbb\xe8\xbb\xe8\xd2\xe5\xd2\xe5\xbb\xe8\xa4\xeb\x8c\xeeu\xf1F\xf7\x00\x00\xd1\x05'
    b'\x8b\x0e\\\x14\x17\x1d\xd1%\x8b.\\4.:\xff?\xe8B\xe8B\xff?\x16=\x16=\\4'
    b'\x8b.\xb9(\xff\x1fE\x17t\x11\xba\x08\xe8\x02\x18\xfdF\xf7u\xf1\x8c\xee\xbb\xe8\xd2\xe5\x01\xe0'
    b'/\xdaG\xd7u\xd1\x8d\xce\xea\xc2\x01\xc0G\xb7\x8d\xae\xd3\xa5\x19\x9dG\x97v\x91\xa4\x8b\xd3\x85'
    b'\xd3\x85\xa4\x8bv\x91G\x97\x19\x9d\xa4\xab\x18\xbd\xa4\xcb\x18\xdd\x8c\xee\x00\x00t\x11\xff\x1f\x8b.'
    b'\x16=\xb9HsQEW\x16]\xff_\xff_\x16]EWsQ\xa2K\xe8B.:\\4'
    b'\xa2+\xd1%\x17\x1dE\x17\\\x14t\x11\xa2\x0b\xba\x08\xba\x08\xd1\x05\xe8\x02\x00\x00\x00\x00\x18\xfd'
    b'/\xfaF\xf7u\xf1\x8c\xee\xa4\xeb\xd2\xe5\xe9\xe2\x01\xe0\x01\xe0\x18\xdd\x01\xe0\xe9\xe2\xbb\xe8\x8c\xee'
    b'u\xf1/\xfa\x00\x00\x8b\x0e\\\x14\x17\x1d\xb9(\\4\x16=\xe8B\xd0E\xa2K\x8bN\xa2K'
    b'\xd0E\xe8B\x16=E7\x8b.\xd1%\x17\x1d\\\x14\xa2\x0b\xe8\x02\x00\x00/\xfa^\xf4\x8c\xee'
    b'\xbb\xe8\xbb\xe8\xd2\xe5\x01\xe0\x18\xdd/\xdaG\xd7\x8d\xce\xbb\xc8\xea\xc2\x18\xbdG\xb7\xbb\xa8\x19\x9d'
    b'0\x9a^\x94\x8d\x8e\xbc\x88\xd3\x85\xbc\x88\x8d\x8e\x8d\x8eG\x97\xd3\xa5G\xb7\xbb\xc8G\xd7\xbb\xe8'
    b'\x00\x00t\x11\xff\x1f\x8b.\xff?\x8bNEW\x16]\xe7b\xd0e\xe7b\xff_-ZEW'
    b'\x8bN\xd0E.:\\4\x8b.\xd1%.\x1a\\\x14t\x11\x8b\x0e\xba\x08\xe8\x02\xe8\x02\xe8\x02'
    b'\x00\x00\x18\xfd/\xfa/\xfaF\xf7^\xf4\x8c\xee\xa4\xeb\xbb\xe8\xe9\xe2\x18\xdd/\xda/\xdaG\xd7'
    b'G\xd7G\xd7\x18\xdd\xe9\xe2\xbb\xe8u\xf1/\xfa\xe8\x02\x8b\x0e.\x1a\xd1%\\4\x16=\xd0E'
    b'\x8bN\\TEW-Z-Z\\T\x8bN\xd0E\x16=s1\xb9(\x17\x1d\\\x14\xa2\x0b'
    b'\x00\x00\x18\xfdF\xf7u\xf1\xa4\xeb\xbb\xe8\xd2\xe5\xe9\xe2\x01\xe0\x18\xdd\x18\xdd/\xda^\xd4\x8d\xce'
    b'\xbb\xc8\xea\xc20\xbau\xb1\xd3\xa5\x19\x9dG\x97\x8d\x8e\xd3\x85\x01\x80\x01\x80\xea\x82\xbc\x88\x8d\x8e'
    b'G\x97\xbb\xa80\xba\xa4\xcb/\xda\x8c\xee\xd1\x05.\x1a\xb9(E7\xb9HEW\xff_\xd0e'
    b'\xb9h\xb9h\xd0e\xff_\x16]\\T\xb9H\x16=E7\x8b.\xd1%.\x1at\x11\xa2\x0b'
    b'\xba\x08\xe8\x02\x00\x00\x18\xfd\x18\xfd\x18\xfd\x18\xfd/\xfaF\xf7F\xf7F\xf7^\xf4\x8c\xee\xa4\xeb'
    b'\xbb\xe8\xd2\xe5\x01\xe0\x18\xdd/\xdaG\xd7G\xd7G\xd7\x18\xdd\xe9\xe2\xd2\xe5\xa4\xeb/\xfa\xd1\x05'
    b'\x8b\x0eE\x17\xd1%\\4\xe8B\xa2KsQ-Z\xe7b\xd0e\xff_\x16]EWsQ'
    b'\xd0E.:s1\xd1%E\x17\xa2\x0b\xe8\x02\x18\xfd^\xf4\x8c\xee\xa4\xeb\xbb\xe8\xbb\xe8\xe9\xe2'
    b'\x01\xe0\x01\xe0\xe9\xe2\x01\xe0/\xdaG\xd7^\xd4\x8d\xce\xea\xc2G\xb7\x8d\xae\xd3\xa5\x19\x9d^\x94'
    b'\xa4\x8b\xd3\x85\x01\x80\x01\x80\xea\x82\xbc\x88\x8d\x8eG\x97\xbb\xa8\x18\xbd\x8d\xce\x01\xe0^\xf4\xba\x08'
    b'\xff\x1fs1\xff?\x8bN-Z\xe7b\xb9h\xa2k\xb9h\xd0e\xff_-ZsQ\xd0E'
    b'.:s1\xb9(\x17\x1d\\\x14\xa2\x0b\xd1\x05\xe8\x02\x00\x00\x00\x00/\xfa/\xfa\x18\xfd\x18\xfd'
    b'/\xfaF\xf7F\xf7F\xf7^\xf4\x8c\xee\xa4\xeb\xd2\xe5\xe9\xe2\x18\xdd'
)

assert len(_wav) == 800
assert len(_wav_bytes) == 1644
assert _sr == 16000


def test_aukit():
    t0 = time.time()
    import aukit
    t = time.time() - t0
    assert t < 5


def test_audio_io():
    from aukit.audio_io import load_wav, save_wav, anything2bytesio, anything2wav, anything2bytes, Dict2Obj, _sr

    out = anything2bytes(_wav, sr=_sr)
    assert len(out) == len(_wav_bytes)

    out = anything2wav(_wav_bytes, sr=_sr)
    assert len(out) == len(_wav)

    my_obj = Dict2Obj({"my_key": "my_value"})
    assert my_obj.my_key == "my_value"


def test_audio_spectrogram():
    from aukit.audio_spectrogram import linear_spectrogram, mel_spectrogram
    from aukit.audio_spectrogram import default_hparams as hparams_spectrogram
    from aukit.audio_spectrogram import linear2mel_spectrogram, mel2linear_spectrogram

    out_linear = linear_spectrogram(_wav, hparams=hparams_spectrogram)
    assert out_linear.shape == (401, 5)

    out_mel = mel_spectrogram(_wav, hparams=hparams_spectrogram)
    assert out_mel.shape == (80, 5)

    out_mel_from_linear = linear2mel_spectrogram(out_linear, hparams=hparams_spectrogram)
    assert out_mel_from_linear.shape == (80, 5)

    out_linear_from_mel = mel2linear_spectrogram(out_mel, hparams=hparams_spectrogram)
    assert out_linear_from_mel.shape == (401, 5)


def test_audio_griffinlim():
    from aukit.audio_griffinlim import inv_linear_spectrogram, inv_linear_spectrogram_tf, inv_mel_spectrogram
    from aukit.audio_griffinlim import default_hparams as hparams_griffinlim
    from aukit.audio_spectrogram import linear_spectrogram, mel_spectrogram

    out_linear = linear_spectrogram(_wav, hparams=hparams_griffinlim)
    out_wav = inv_linear_spectrogram(out_linear, hparams=hparams_griffinlim)
    assert out_wav.shape == (800,)

    out_mel = mel_spectrogram(_wav, hparams=hparams_griffinlim)
    out_wav = inv_mel_spectrogram(out_mel, hparams=hparams_griffinlim)
    assert out_wav.shape == (800,)


def test_audio_player():
    from aukit.audio_player import play_audio

    play_audio(_wav, sr=_sr)
    play_audio(_wav_bytes, sr=_sr)


if __name__ == "__main__":
    print(__file__)
