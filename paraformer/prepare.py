#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import re
import soundfile as sf
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# 设置并行处理的核心数
NUM_CORES = multiprocessing.cpu_count()
# 设置更大的批处理大小
BATCH_SIZE = 1000

def process_text(file_path):
    """处理文本文件，移除时间戳并合并文本行"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        timestamp_pattern = r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
        if re.search(timestamp_pattern, content):
            content = re.sub(timestamp_pattern, "", content)
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            text = " ".join(lines)
        else:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            text = " ".join(lines)
        return text.strip()

def process_audio_batch(audio_files):
    """处理一批音频文件"""
    results = {}
    for line in audio_files:
        lines = line.strip().split(maxsplit=1)
        if not lines:
            continue
        key = lines[0]
        audio_path = lines[1] if len(lines) > 1 else ""
        if os.path.exists(audio_path):
            try:
                waveform, sr = sf.read(audio_path)
            except Exception:
                continue
            sample_num = len(waveform)
            # 这里计算的单位为0.1秒
            context_len = int(sample_num * 1000 / sr / 10)
            results[key] = {"source": audio_path, "source_len": context_len}
    return results

def process_text_batch(text_lines):
    """处理一批文本文件"""
    results = {}
    for line in text_lines:
        lines = line.strip().split(maxsplit=1)
        if not lines:
            continue
        key = lines[0]
        text = lines[1] if len(lines) > 1 else ""
        # 这里计算的单位为词数或字符数，可根据需求调整
        context_len = len(text.split()) if " " in text else len(text)
        results[key] = {"target": text, "target_len": context_len}
    return results

def parse_files_parallel(file_path, process_func):
    """并行处理文件"""
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"\n处理文件 {file_path}")
    print(f"总计 {total_lines} 条记录")
    
    # 将数据分成批次
    batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
    total_batches = len(batches)
    
    with tqdm(total=total_lines, desc="处理进度", unit="条") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = [executor.submit(process_func, batch) for batch in batches]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                results.update(future.result())
                pbar.update(min(BATCH_SIZE, len(lines) - (i-1)*BATCH_SIZE))
                pbar.set_postfix({"批次": f"{i}/{total_batches}"})
    
    return results

def split_and_save_jsonl(json_dict, data_type_list, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    """将数据分割为训练集、验证集和测试集，并保存为jsonl文件"""
    all_keys = list(json_dict[data_type_list[0]].keys())
    random.shuffle(all_keys)
    total = len(all_keys)
    train_split = int(total * train_ratio)
    val_split = int(total * (train_ratio + val_ratio))
    train_keys = all_keys[:train_split]
    val_keys = all_keys[train_split:val_split]
    test_keys = all_keys[val_split:]
    
    def write_batch(keys, output_file):
        with open(output_file, "w", encoding='utf-8') as f:
            for key in keys:
                jsonl_line = {"key": key}
                for data_file in data_type_list:
                    if key in json_dict[data_file]:
                        jsonl_line.update(json_dict[data_file][key])
                f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
    
    write_batch(train_keys, train_file)
    write_batch(val_keys, val_file)
    write_batch(test_keys, test_file)
    
    return len(train_keys), len(val_keys), len(test_keys)

def gen_jsonl_from_wav_text_list(path, data_type_list=("source", "target"), train_file=None, val_file=None, test_file=None):
    """从wav.scp和text文件生成jsonl文件"""
    print(f"\n使用 {NUM_CORES} 个CPU核心并行处理")
    
    print("\n1. 处理音频文件...")
    json_dict = {
        "source": parse_files_parallel(path[0], process_audio_batch),
    }
    
    print("\n2. 处理文本文件...")
    json_dict["target"] = parse_files_parallel(path[1], process_text_batch)
    
    print("\n3. 生成训练集、验证集和测试集...")
    train_count, val_count, test_count = split_and_save_jsonl(
        json_dict,
        data_type_list,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"\n处理完成：")
    print(f"- 训练集：{train_count} 样本")
    print(f"- 验证集：{val_count} 样本")
    print(f"- 测试集：{test_count} 样本")
    print(f"- 总计：{train_count + val_count + test_count} 样本")
    
    return train_count, val_count, test_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成用于语音识别训练的jsonl文件")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    parser.add_argument("--scp_file", help="音频列表文件路径")
    parser.add_argument("--text_file", help="文本列表文件路径")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    args = parser.parse_args()
    
    # 如果未指定scp_file和text_file，则使用默认路径
    scp_file = args.scp_file or os.path.join(args.output_dir, "train_wav.scp")
    text_file = args.text_file or os.path.join(args.output_dir, "train_text.txt")
    
    # 检查文件是否存在
    if not os.path.exists(scp_file):
        raise FileNotFoundError(f"音频列表文件不存在: {scp_file}")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"文本列表文件不存在: {text_file}")
    
    # 设置输出文件路径
    train_jsonl = os.path.join(args.output_dir, "train.jsonl")
    val_jsonl = os.path.join(args.output_dir, "val.jsonl")
    test_jsonl = os.path.join(args.output_dir, "test.jsonl")
    
    # 生成jsonl文件
    train_count, val_count, test_count = gen_jsonl_from_wav_text_list(
        path=[scp_file, text_file],
        data_type_list=["source", "target"],
        train_file=train_jsonl,
        val_file=val_jsonl,
        test_file=test_jsonl,
    )
    
    return 0

if __name__ == "__main__":
    exit(main()) 