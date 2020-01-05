
## aukit
audio toolkit: 语音和频谱处理的工具箱。

### 安装

```
pip install aukit
```

- 注意
    * 如果安装过程中依赖报错，则单独另外安装依赖后，再安装aukit。
    * 容易安装报错的依赖包：tensorflow,pyaudio,sounddevice。
    * TensorFlow<=1.13.1
    * pyaudio暂不支持python37以上版本直接pip安装，需要下载whl文件安装，下载路径：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### 版本
v1.3.1

### audio_cli
命令行，播放音频。

### audio_changer
变声器，变高低音，变语速，变萝莉音，回声。

### audio_editor
语音编辑，切分音频，去除语音中的较长静音，去除语音首尾静音。

### audio_griffinlim
griffinlim声码器，线性频谱转语音，梅尔频谱转语音，TensorFlow版本转语音。

### audio_io
语音保存、读取。

### audio_noise_remover
语音降噪，降低环境噪声。
todo list:
1. 添加自定义噪声样本。
2. 添加可设置降噪阈值等参数。

### audio_normalizer
语音正则化，去除语音中音量低的部分，标准化音量。

### audio_player
语音播放，传入文件名播放，播放wave数据，播放bytes数据。

### audio_spectrogram
语音频谱，语音转线性频谱，语音转梅尔频谱。

### audio_tuner
语音调整，调整语速，调整音高。
