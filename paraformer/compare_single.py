#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import time
import os
import difflib

def calculate_cer(ref, hyp):
    """计算字错率 (Character Error Rate)"""
    ref, hyp = str(ref), str(hyp)
    d = difflib.SequenceMatcher(None, ref, hyp)
    errors = len(ref) + len(hyp) - 2 * d.ratio() * len(ref)
    return (float(errors) / len(ref) if len(ref) > 0 else float('inf')) * 100

def evaluate_model(model_path, model_file=None, audio_file=None, reference_text=None, hotword=None):
    """评估单个模型的识别效果"""
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"错误：模型路径不存在 - {model_path}")
    
    print(f"\n正在评估模型: {model_path}")

    # 检查模型文件
    if model_file and not os.path.exists(os.path.join(model_path, model_file)):
        raise ValueError(f"错误：模型文件不存在 - {os.path.join(model_path, model_file)}")
    
    # 加载模型
    try:
        if model_file:
            model = AutoModel(
                model=model_path,
                model_file=os.path.join(model_path, model_file),
                device="cuda",
                disable_update=False
            )
        else:
            # 检查VAD模型路径
            vad_path = "/media/fl01/data01/WorkSpace/FunASR/model/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            if not os.path.exists(vad_path):
                raise ValueError(f"错误：VAD模型路径不存在 - {vad_path}")
                
            model = AutoModel(
                model=model_path,
                vad_model=vad_path,
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                disable_update=True
            )
    except Exception as e:
        raise ValueError(f"模型加载失败：{str(e)}")
    
    # 获取音频文件路径
    audio_path = audio_file if audio_file else os.path.join(model.model_path, "example", "zh.wav")
    print(f"使用音频文件: {audio_path}")
    
    start_time = time.time()
    
    # 进行识别
    generate_params = {
        "input": audio_path,
        "cache": {},
        "language": "zh",
        "use_itn": True,
        "batch_size_s": 60,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    
    # 如果提供了热词，添加到参数中
    if hotword:
        generate_params["hotword"] = hotword
    
    res = model.generate(**generate_params)
    
    inference_time = time.time() - start_time
    text = rich_transcription_postprocess(res[0]["text"])
    
    # 如果提供了参考文本，计算错误率
    metrics = {}
    if reference_text:
        metrics['cer'] = calculate_cer(reference_text, text)
    
    return text, inference_time, metrics

def main():
    # 指定单个测试音频和文本文件
    audio_file = "/media/fl01/data01/WorkSpace/srtprocessor/wav/0104.wav"  # 请修改为实际的音频文件路径
    text_file = "/media/fl01/data01/WorkSpace/srtprocessor/0104.txt"   # 请修改为实际的文本文件路径
    
    print("开始模型对比测试...")
    
    # 读取参考文本
    with open(text_file, 'r', encoding='utf-8') as f:
        reference_text = f.read().strip()
    
    print(f"测试音频: {audio_file}")
    
    # 测试预训练模型（不设置热词）
    pretrained_text, pretrained_time, pretrained_metrics = evaluate_model(
        "/media/fl01/data01/WorkSpace/FunASR/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        audio_file=audio_file,
        reference_text=reference_text,
        #hotword=""
    )
    
    # 测试微调后的模型（设置热词）
    finetuned_text, finetuned_time, finetuned_metrics = evaluate_model(
        "./outputs",
        model_file="model.pt.best",
        audio_file=audio_file,
        reference_text=reference_text,
        #hotword=""  # 为微调模型单独设置热词
    )
    
    # 打印测试结果
    print("\n=== 测试结果 ===")
    print(f"参考文本: {reference_text}")
    print(f"\n预训练模型:")
    print(f"识别结果: {pretrained_text}")
    print(f"字错率(CER): {pretrained_metrics['cer']:.2f}%")
    print(f"识别耗时: {pretrained_time:.2f}秒")
    
    print(f"\n微调模型:")
    print(f"识别结果: {finetuned_text}")
    print(f"字错率(CER): {finetuned_metrics['cer']:.2f}%")
    print(f"识别耗时: {finetuned_time:.2f}秒")

if __name__ == "__main__":
    main()
