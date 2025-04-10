**基础文本处理（FunTextProcessing）**
==========================

### 简介

FunTextProcessing 是一个用于自动语音识别（ASR）基础文本处理的 Python 工具包，包含文本处理、逆文本处理（ITN）、数字转文字（num2words）等功能。该工具包隶属于 `FunASR` 项目。

### 亮点

- 支持逆文本处理（ITN）、文本处理（TN）及数字转文字（num2words）功能。
- 多语言支持：覆盖 10+ 种语言的逆文本处理、5 种语言的文本处理及 50+ 种语言的数字转文字。

### 示例
#### 逆文本处理（ITN）
给定文本输入（如语音识别结果），使用 `fun_text_processing/inverse_text_normalization/inverse_normalize.py` 可输出逆文本处理结果。参考以下示例脚本：

```
test_file=fun_text_processing/inverse_text_normalization/id/id_itn_test_input.txt

python fun_text_processing/inverse_text_normalization/inverse_normalize.py --input_file $test_file --cache_dir ./itn_model/ --output_file output.txt --language=id
```

### 致谢
1. 代码大量参考了 [NeMo](https://github.com/NVIDIA/NeMo) 项目。
2. 中文逆文本处理功能参考了 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) 的实现。
3. 部分语言的数字转文字功能借用了 [num2words](https://pypi.org/project/num2words/) 库的代码。

### 许可证

本项目基于 [MIT 许可证](https://opensource.org/licenses/MIT) 开源。FunTextProcessing 还包含基于其他开源许可证的第三方组件及修改代码。%             