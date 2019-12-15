#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
语音处理工具箱。
"""

from setuptools import setup, find_packages

install_requires = ['webrtcvad', 'pydub', 'lws', 'scipy', 'numpy', 'pyaudio', 'librosa', 'pyworld', 'tensorflow']

with open("README.md", "rt", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="aukit",
    version="1.0.0",
    author="kuangdd",
    author_email="kqhyj@163.com",
    description="audio tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KuangDD/audiotools",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=install_requires,  # 指定项目最低限度需要运行的依赖项
    python_requires='>=3.5',  # python的依赖关系
    package_data={
        'info': ['README.md', 'requirements.txt'],
    },  # 包数据，通常是与软件包实现密切相关的数据
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'pla = aukit.audio_player:play_audio_cmd'
        ]
    }
)

if __name__ == "__main__":
    print(__file__)
