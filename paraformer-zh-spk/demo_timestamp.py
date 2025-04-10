#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
import os
import datetime

def format_timestamp(ms):
    """将毫秒转换为 HH:MM:SS 格式"""
    seconds = ms / 1000  # 转换为秒
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

model_pwd_dir = "model"
input_path = "/media/fl01/data01/Data/manual171_2/clean/data/wyc/manual171_2_0017.wav"

# 获取输入文件的名称（不含扩展名）
base_name = os.path.splitext(os.path.basename(input_path))[0]
# 创建输出目录（与输入文件相同目录）
output_dir = os.path.dirname(input_path)
# 构建输出文件路径
output_path = os.path.join(output_dir, f"{base_name}.txt")

model = AutoModel(
    disable_update=True,
    device="cuda",
    model=f"{model_pwd_dir}/paraformer_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model=f"{model_pwd_dir}/vad_models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model=f"{model_pwd_dir}/punc_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    spk_model=f"{model_pwd_dir}/spk_models/speech_campplus_sv_zh-cn_16k-common",
    timestamp=True,  # 需要开启时间戳
)

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
            speaker = f"spk{sentence.get('spk', 'unknown')}"
            text = sentence.get('text', '')
            start_time = format_timestamp(sentence.get('start', 0))
            end_time = format_timestamp(sentence.get('end', 0))
            
            formatted_line = f"{start_time} - {end_time}  {speaker}   {text}"
            formatted_results.append(formatted_line)
    elif 'timestamp' in result:
        # 如果没有 sentence_info，则使用原来的处理方式
        speaker = f"spk{result.get('spk', 'unknown')}"
        text = result.get('text', '')
        
        for ts in result['timestamp']:
            start_time = format_timestamp(ts[0])
            end_time = format_timestamp(ts[1])
            
            formatted_line = f"{start_time} - {end_time}  {speaker}   {text}"
            formatted_results.append(formatted_line)
    else:
        # 如果没有时间戳信息，直接添加文本
        formatted_results.append(result.get('text', ''))

# 打印格式化的结果
print("识别结果格式化输出:")
for line in formatted_results:
    print(line)

# 保存到文件
with open(output_path, 'w', encoding='utf-8') as f:
    for line in formatted_results:
        f.write(line + '\n')

print(f"\n识别结果已保存到：{output_path}")
