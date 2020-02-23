#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
import time


def test_aukit():
    t0 = time.time()
    import aukit
    t = time.time() - t0
    assert t < 3


if __name__ == "__main__":
    print(__file__)
