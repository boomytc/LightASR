## 🔊 LightASR

## 📌 项目简介

**LightASR** 是一个基于开源项目 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 精简开发的语音识别系统，专注于以下功能：

- 🎙 支持语音识别（可选模型）  
- 📊 多模型识别结果对比（准确率评估）  
- 🖥 提供简洁 GUI（基于 PyQt6）和 Web UI（基于 Streamlit）

# 安装依赖
```bash
pip install -r requirements.txt
```

# 下载离线依赖包（可选）
```bash
pip download -r ./requirements.txt -d ./offline-packages
```

# 安装下载的离线依赖包（可选）
```bash
pip install --no-index --find-links=offline-packages -r requirements.txt
```