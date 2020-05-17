
## aukit
audio toolkit: 语音和频谱处理的工具箱。

### 安装

```
pip install aukit
```

- 注意
    * 可能需另外安装的依赖包：tensorflow, pyaudio, sounddevice。
    * tensorflow<=1.13.1
    * pyaudio暂不支持python37以上版本直接pip安装，需要下载whl文件安装，下载路径：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
    * sounddevice依赖pyaudio。
    * aukit的默认音频采样率为16k。

### 版本
v1.3.10

### audio_cli
命令行，播放音频，去除背景噪声。

### audio_changer
变声器，变高低音，变语速，变萝莉音，回声。

### audio_editor
语音编辑，切分音频，去除语音中的较长静音，去除语音首尾静音，设置采样率，设置通道数。
切分音频，去除静音，去除首尾静音输入输出都支持wav格式。
语音编辑功能基于pydub的方法，增加了数据格式支持。

### audio_griffinlim
griffinlim声码器，线性频谱转语音，梅尔频谱转语音，TensorFlow版本转语音，梅尔频谱和线性频谱相互转换。

### audio_io
语音IO，语音保存、读取，语音格式转换，支持【.】操作符的字典。

### audio_noise_remover
语音降噪，降低环境噪声。

### audio_normalizer
语音正则化，去除音量低的音频段，调节音量。
语音正则化方法基于VAD的方法。

### audio_player
语音播放，传入文件名播放，播放wave数据，播放bytes数据。

### audio_spectrogram
语音频谱，语音转线性频谱，语音转梅尔频谱。

### audio_tuner
语音调整，调整语速，调整音高。

### audio_world
world声码器，提取语音的基频、频谱包络和非周期信号，频谱转为语音。调音高，调机器人音。
