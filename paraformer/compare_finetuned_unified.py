#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import time
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import re
import argparse
import sys
import contextlib
import io
import subprocess
import psutil
from datetime import datetime
import gc
import json

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    print("请先安装 funasr：pip install funasr")
    sys.exit(1)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """临时禁止所有标准输出和标准错误的上下文管理器"""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    null = io.StringIO()
    sys.stdout = null
    sys.stderr = null
    
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def remove_punctuation(text):
    """移除文本中的标点符号"""
    punctuation = r'[，。！？、；：""''（）《》【】\.,!?;:\'\"\(\)\[\]，。！？、；：""''（）《》【】]'
    return re.sub(punctuation, '', text)


def calculate_cer(ref, hyp):
    """计算字错率 (Character Error Rate)，忽略标点符号"""
    # 首先移除参考文本和识别文本中的标点符号
    ref = remove_punctuation(str(ref))
    hyp = remove_punctuation(str(hyp))
    
    # 处理参考文本为空的特殊情况
    if len(ref) == 0:
        return float('inf')
    
    # 使用动态规划计算编辑距离
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化第一行和第一列
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充dp表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # 替换
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1     # 插入
                )
    
    # 计算字错率
    error_count = dp[m][n]
    cer = (error_count / len(ref)) * 100
    
    return cer


def get_gpu_memory_info():
    """获取所有GPU的显存信息"""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd.split(), universal_newlines=True)
        
        gpu_info = []
        for line in output.strip().split("\n"):
            index, total, used, free = map(float, line.split(","))
            gpu_info.append({
                'index': int(index),
                'total': total,
                'used': used,
                'free': free,
                'utilization': (used / total) * 100
            })
        return gpu_info
    except:
        return None


def process_audio(model, audio_path, cache):
    """处理单个音频文件"""
    generate_params = {
        "input": audio_path,
        "cache": cache,
        "language": "zh",
        "use_itn": True,
        "batch_size_s": 16,
        "merge_vad": True,
        "merge_length_s": 15,
    }
    torch.cuda.empty_cache()
    
    with suppress_stdout_stderr():
        with torch.no_grad():
            res = model.generate(**generate_params)
    
    # 添加结果检查
    if not res or len(res) == 0:
        raise ValueError(f"模型未能生成有效结果: {audio_path}")
    
    text = rich_transcription_postprocess(res[0]["text"]).replace(" ", "")
    if not text.strip():
        raise ValueError(f"模型生成了空文本: {audio_path}")
    
    del res
    torch.cuda.empty_cache()
    
    return text, generate_params["cache"]


def get_test_pairs_from_jsonl(jsonl_path):
    """从jsonl文件获取测试音频和对应的文本配对"""
    test_pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            test_pairs.append({
                'audio': data['source'],
                'text': data['target']
            })
    return test_pairs


def get_test_pairs_from_dirs(wav_dir, txt_dir):
    """
    从目录获取测试音频和对应的文本配对（包含子文件夹）
    以txt文件名为主，如果wav文件名包含txt文件名（不含扩展名），则认为匹配成功
    """
    test_pairs = []
    
    for root, _, files in os.walk(txt_dir):
        for txt_file in files:
            if not txt_file.endswith('.txt'):
                continue
            rel_path = os.path.relpath(root, txt_dir)
            txt_base_name = os.path.splitext(txt_file)[0]
            txt_full_path = os.path.join(root, txt_file)
            
            wav_search_dir = os.path.join(wav_dir, rel_path)
            if not os.path.exists(wav_search_dir):
                continue
                
            with open(txt_full_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()
            
            matched_wavs = []
            for wav_file in os.listdir(wav_search_dir):
                if wav_file.endswith('.wav') and txt_base_name in wav_file:
                    matched_wavs.append(os.path.join(wav_search_dir, wav_file))
            
            for wav_path in matched_wavs:
                test_pairs.append({
                    'audio': wav_path,
                    'text': reference_text
                })
    
    return test_pairs


class ResourceMonitor:
    """资源监控器"""
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.last_update = 0
        self.update_interval = 2
        self.resource_progress = tqdm(
            total=0,
            bar_format='{desc}',
            position=0,
            leave=True
        )
    
    def get_gpu_info(self):
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            return "GPU信息获取失败"
        
        gpu_str = "GPU: "
        for i, info in enumerate(gpu_info[:self.num_gpus]):
            gpu_str += f"[{i}: {info['used']:.0f}/{info['total']:.0f}MB {info['utilization']:.1f}%] "
        return gpu_str
    
    def get_cpu_info(self):
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        return f"CPU: {cpu_percent:.1f}% | 内存: {mem.used/1024/1024/1024:.1f}/{mem.total/1024/1024/1024:.1f}GB {mem.percent:.1f}%"
    
    def get_resource_info(self):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        gpu_info = self.get_gpu_info()
        cpu_info = self.get_cpu_info()
        self.resource_progress.set_description_str(f"{gpu_info} | {cpu_info}")


def process_gpu_batch(gpu_id, model_paths, data_queue, result_queue):
    """单GPU进程处理函数"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # 加载模型
    with suppress_stdout_stderr():
        models = {}
        for model_name, m_path in list(model_paths.items()):
            try:
                vad_path = "/media/fl01/data01/WorkSpace/FunASR/model/vad_models/speech_fsmn_vad_zh-cn-16k-common-pytorch/"
                models[model_name] = AutoModel(
                    model=m_path,
                    vad_model=vad_path,
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=f"cuda:{gpu_id}",
                    disable_update=True
                )
            except Exception as e:
                print(f"GPU {gpu_id} 加载模型 {model_name} 失败: {str(e)}")
                return

    process_cache = {}
    
    while True:
        batch_data = data_queue.get()
        if batch_data is None:
            break

        batch_results = []
        for pair in batch_data:
            single_result = {'pair': pair, 'results': {}}
            
            model_keys = list(models.keys())
            for model_name in model_keys:
                model = models[model_name]
                try:
                    cache_key = f"{model_name}_{pair['audio']}"
                    cache = process_cache.get(cache_key, {})
                    
                    start_time = time.time()
                    text, updated_cache = process_audio(model, pair['audio'], cache)
                    process_time = time.time() - start_time
                    
                    # 更新缓存
                    process_cache[cache_key] = updated_cache
                    
                    # 计算CER
                    cer = calculate_cer(pair['text'], text)
                    
                    single_result['results'][model_name] = {
                        'text': text,
                        'time': process_time,
                        'metrics': {
                            'cer': cer
                        },
                        'success': True
                    }
                    
                except Exception as e:
                    print(f"处理音频 {pair['audio']} 失败: {str(e)}")
                    single_result['results'][model_name] = {
                        'error': str(e),
                        'success': False
                    }
            
            batch_results.append(single_result)
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 返回结果
        result_queue.put(batch_results)
    
    # 清理模型
    for model_name in list(models.keys()):
        del models[model_name]
    models.clear()
    process_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def create_processes(model_paths, num_procs_per_gpu=1):
    """为每块 GPU 启动固定数量的进程"""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("未检测到可用的GPU")
    
    # 创建进程间通信的队列
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 为每个GPU分配进程
    processes = []
    processes_per_gpu = {}
    
    for gpu_id in range(num_gpus):
        processes_per_gpu[gpu_id] = num_procs_per_gpu
        for _ in range(num_procs_per_gpu):
            p = mp.Process(
                target=process_gpu_batch,
                args=(gpu_id, model_paths, data_queue, result_queue)
            )
            p.daemon = True
            p.start()
            processes.append(p)
    
    return processes_per_gpu, data_queue, result_queue, processes


def main():
    parser = argparse.ArgumentParser(description='比较多个模型的识别效果')
    
    # 添加数据输入方式选择参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--jsonl', help='包含测试数据的jsonl文件路径')
    input_group.add_argument('--wav_txt_dirs', nargs=2, metavar=('WAV_DIR', 'TXT_DIR'), 
                           help='音频文件目录和文本文件目录路径')
    
    # 其他参数
    parser.add_argument('--models', required=True, nargs='+', 
                      help='模型路径列表，格式: 模型名称 模型路径 [模型名称 模型路径 ...]')
    parser.add_argument('--result', required=True, help='结果保存目录路径')
    parser.add_argument('--proc_per_gpu', type=int, default=1, help='每块GPU启动的进程数')
    parser.add_argument('--output_filename', default='results.txt', help='结果输出文件名 (默认: results.txt)') # 新增参数
    args = parser.parse_args()
    
    # 创建结果保存目录
    os.makedirs(args.result, exist_ok=True)
    # 使用 args.output_filename 构建结果文件路径
    result_file = os.path.join(args.result, args.output_filename) 
    
    # 解析模型参数
    if len(args.models) % 2 != 0:
        raise ValueError("模型参数必须是名称和路径的配对")
    model_paths = {args.models[i]: args.models[i+1] for i in range(0, len(args.models), 2)}
    
    print("开始模型对比测试...")
    print(f"要比较的模型: {list(model_paths.keys())}")
    
    # 启动子进程
    processes_per_gpu, data_queue, result_queue, processes = create_processes(
        model_paths, num_procs_per_gpu=args.proc_per_gpu
    )
    total_processes = sum(processes_per_gpu.values())
    
    # 打印 GPU 资源分配情况
    print("\n=== GPU 资源分配情况 ===")
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        for gpu_id, num_proc in processes_per_gpu.items():
            if gpu_id < len(gpu_info):
                gi = gpu_info[gpu_id]
                print(f"GPU {gpu_id}:")
                print(f"  - 总显存: {gi['total']:.0f}MB")
                print(f"  - 已用显存: {gi['used']:.0f}MB ({gi['utilization']:.1f}%)")
                print(f"  - 空闲显存: {gi['free']:.0f}MB")
                print(f"  - 分配进程数: {num_proc}")
    print(f"总进程数: {total_processes}\n")
    
    # 等待一下确保进程都启动完成
    time.sleep(2)
    
    # 读取测试文件对
    if args.jsonl:
        print(f"从JSONL文件加载测试数据: {args.jsonl}")
        test_pairs = get_test_pairs_from_jsonl(args.jsonl)
    else:
        wav_dir, txt_dir = args.wav_txt_dirs
        print(f"从目录加载测试数据: 音频目录={wav_dir}, 文本目录={txt_dir}")
        test_pairs = get_test_pairs_from_dirs(wav_dir, txt_dir)
    
    total_files = len(test_pairs)
    print(f"共读取到音频文件数: {total_files}")
    
    # 将数据分批
    batch_size = 8
    batches = [test_pairs[i:i + batch_size] for i in range(0, total_files, batch_size)]
    
    # 初始化资源监控器
    resource_monitor = ResourceMonitor(num_gpus=torch.cuda.device_count())
    
    # 修改结果收集部分
    results = {name: {"cer_sum": 0, "time_sum": 0, "success_count": 0} for name in model_paths.keys()}
    processed_count = 0
    
    print("\n开始处理音频文件...")
    start_all_time = time.time()
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            progress_bar = tqdm(
                total=total_files,
                desc="处理进度",
                ncols=100,
                position=1,
                leave=True
            )
            
            # 分发数据并处理结果
            for batch in batches:
                data_queue.put(batch)
                
                try:
                    batch_results = result_queue.get(timeout=300)
                    for single_result in batch_results:
                        pair = single_result['pair']
                        processed_count += 1
                        
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(
                            f"文件: {os.path.basename(pair['audio'])} | "
                            f"时间: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        
                        resource_monitor.get_resource_info()
                        
                        f.write(f"\n[{processed_count}/{total_files}] 文件: {os.path.basename(pair['audio'])}\n")
                        f.write(f"参考文本: {pair['text']}\n")
                        f.write(f"参考文本(无标点): {remove_punctuation(pair['text'])}\n")
                        
                        for model_name, model_result in single_result['results'].items():
                            if model_result.get('success', False):
                                f.write(f"{model_name} 识别结果: {model_result['text']}\n")
                                cer_val = model_result['metrics']['cer']
                                f.write(f"{model_name} CER: {cer_val:.2f}%\n")
                                
                                results[model_name]["cer_sum"] += cer_val
                                results[model_name]["time_sum"] += model_result['time']
                                results[model_name]["success_count"] += 1
                        
                        f.write("-" * 80 + "\n")
                        f.flush()
                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    continue
            
            progress_bar.close()
        
        # 统计并输出最终结果
        total_time = time.time() - start_all_time
        print(f"\n结果已保存至：{result_file}")
        print("\n=== 测试结果统计 ===")
        print(f"总测试数据数量: {total_files}")
        
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write("\n\n=== 测试结果统计 ===\n")
            f.write(f"总测试数据数量: {total_files}\n")
            f.write(f"整体任务总耗时: {total_time:.2f} 秒\n")

            for model_name, stats in results.items():
                success_count = stats["success_count"]
                if success_count > 0:
                    avg_cer = stats["cer_sum"] / success_count
                    avg_time = stats["time_sum"] / success_count
                    summary = (
                        f"\n{model_name}:\n"
                        f"成功处理数量: {success_count}/{total_files}\n"
                        f"平均字错率(CER): {avg_cer:.2f}%\n"
                        f"平均识别耗时(单条): {avg_time:.2f}秒\n"
                    )
                    print(summary)
                    f.write(summary)
    
    finally:
        print("\n正在清理进程...")
        # 发送退出信号
        for _ in range(total_processes):
            data_queue.put(None)
        
        # 等待进程结束
        for p in processes:
            if p.is_alive():
                p.join(timeout=30)
                if p.is_alive():
                    print(f"进程 {p.pid} 未能正常退出，强制终止")
                    p.terminate()
        
        # 清空队列
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
            except:
                pass
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                pass
        
        print("所有进程已清理完毕")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
