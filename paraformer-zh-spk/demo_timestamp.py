#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
import os
import datetime
import argparse
import sys

def format_timestamp(ms):
    """将毫秒转换为 HH:MM:SS,ms 格式"""
    seconds = int(ms / 1000)  # 整数秒
    milliseconds = int(ms % 1000)  # 毫秒部分
    
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='使用FunASR进行语音识别并提取时间戳')
parser.add_argument('input_path', type=str, help='输入音频文件的路径')
parser.add_argument('--asr_model', type=str, required=True, help='ASR模型路径')
parser.add_argument('--device', type=str, default="cuda", help='运行设备，可选cuda或cpu')

args = parser.parse_args()

# 检查输入文件是否存在
if not os.path.exists(args.input_path):
    print(f"错误: 输入文件 '{args.input_path}' 不存在!")
    sys.exit(1)

input_path = args.input_path

# 获取输入文件的名称（不含扩展名）
base_name = os.path.splitext(os.path.basename(input_path))[0]
# 创建输出目录（与输入文件相同目录）
output_dir = os.path.dirname(input_path)
# 构建输出文件路径
output_path = os.path.join(output_dir, f"{base_name}.txt")

# 固定使用相对路径models目录
model_pwd_dir = "models"

# 准备模型参数
model_params = {
    "disable_update": True,
    "device": args.device,
    "model": args.asr_model,
    "vad_model": f"{model_pwd_dir}/vad_models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_model": f"{model_pwd_dir}/punc_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "spk_model": f"{model_pwd_dir}/spk_models/speech_campplus_sv_zh-cn_16k-common",
    "timestamp": True,  # 需要开启时间戳
}

print(f"加载模型参数: {model_params}")
model = AutoModel(**model_params)

print(f"正在处理音频文件: {input_path}")
res = model.generate(
    input=input_path,
)
print("识别结果结构：", res)  # 打印结果，查看实际的数据结构

# 将结果保存到文件，格式化输出
formatted_results = []  # 存储格式化后的结果

for result in res:
    if 'sentence_info' in result:
        # 使用 sentence_info 中的分句信息
        for sentence in result['sentence_info']:
            # 将说话人编号加1，从1开始
            spk_value = sentence.get('spk')
            if spk_value is not None and isinstance(spk_value, int):
                speaker = f"spk{spk_value + 1}"
            else:
                speaker = "spk1"  # 默认为spk1
            
            # 去除句尾标点符号
            text = sentence.get('text', '').rstrip(',.。，!！?？')
            start_time = format_timestamp(sentence.get('start', 0))
            end_time = format_timestamp(sentence.get('end', 0))
            
            formatted_line = f"{start_time} --> {end_time}  {speaker}   {text}"
            formatted_results.append(formatted_line)
    elif 'timestamp' in result:
        # 如果没有 sentence_info，则使用原来的处理方式
        # 将说话人编号加1，从1开始
        spk_value = result.get('spk')
        if spk_value is not None and isinstance(spk_value, int):
            speaker = f"spk{spk_value + 1}"
        else:
            speaker = "spk1"  # 默认为spk1
            
        # 去除句尾标点符号
        text = result.get('text', '').rstrip(',.。，!！?？')
        
        for ts in result['timestamp']:
            start_time = format_timestamp(ts[0])
            end_time = format_timestamp(ts[1])
            
            formatted_line = f"{start_time} --> {end_time}  {speaker}   {text}"
            formatted_results.append(formatted_line)
    else:
        # 如果没有时间戳信息，直接添加文本，同样去除句尾标点
        formatted_results.append(result.get('text', '').rstrip(',.。，!！?？'))

# 打印格式化的结果
print("识别结果格式化输出:")
for line in formatted_results:
    print(line)

# 保存到文件
with open(output_path, 'w', encoding='utf-8') as f:
    for line in formatted_results:
        f.write(line + '\n')

print(f"\n识别结果已保存到：{output_path}")
