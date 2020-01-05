#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
语音处理工具箱。
"""

from setuptools import setup, find_packages

from aukit import __version__ as aukit_version


def create_readme():
    from aukit import __doc__, version_doc, cli_doc, changer_doc, editor_doc, griffinlim_doc, io_doc, noise_remover_doc
    from aukit import normalizer_doc, player_doc, spectrogram_doc, tuner_doc
    docs = []
    with open("README.md", "wt", encoding="utf8") as fout:
        for doc in [__doc__, version_doc, cli_doc, changer_doc, editor_doc, griffinlim_doc, io_doc, noise_remover_doc,
                    normalizer_doc, player_doc, spectrogram_doc, tuner_doc]:
            fout.write(doc)
            docs.append(doc)
    return "".join(docs)


aukit_doc = create_readme()
install_requires = [w.strip() for w in open("requirements.txt", encoding="utf8") if w.strip()]
print(install_requires)
setup(
    name="aukit",
    version=aukit_version,
    author="kuangdd",
    author_email="kuangdd@foxmail.com",
    description="audio toolkit",
    long_description=aukit_doc,
    long_description_content_type="text/markdown",
    url="https://github.com/KuangDD/aukit",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=install_requires,  # 指定项目最低限度需要运行的依赖项
    python_requires='>=3.5',  # python的依赖关系
    package_data={
        'info': ['README.md', 'requirements.txt'],
    },  # 包数据，通常是与软件包实现密切相关的数据
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'aup = aukit.audio_cli:play_audio_cli'
        ]
    }
)

if __name__ == "__main__":
    print(__file__)
