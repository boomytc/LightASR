import gradio as gr
from funasr import AutoModel
import os
import datetime
import configparser

# 辅助函数，参考自 demo_timestamp.py
def format_timestamp(ms):
    """将毫秒转换为 HH:MM:SS,ms 格式"""
    seconds = int(ms / 1000)
    milliseconds = int(ms % 1000)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# --- 从 model.conf 加载模型配置 ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'model.conf')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件 'model.conf' 未找到: {config_path}")
config.read(config_path)

# 获取各种模型列表
asr_models = dict(config['asr_models_dir']) if 'asr_models_dir' in config else {}
vad_models = dict(config['vad_models_dir']) if 'vad_models_dir' in config else {}
punc_models = dict(config['punc_models_dir']) if 'punc_models_dir' in config else {}
spk_models = dict(config['spk_models_dir']) if 'spk_models_dir' in config else {}

# 默认模型配置 (使用配置文件中的第一个作为默认值)
default_asr = list(asr_models.keys())[0] if asr_models else None
default_vad = list(vad_models.keys())[0] if vad_models else None
default_punc = list(punc_models.keys())[0] if punc_models else None
default_spk = list(spk_models.keys())[0] if spk_models else None

# --- 全局模型缓存 ---
model_cache = {}

def get_model(device, asr_model_name, vad_model_name, punc_model_name, spk_model_name):
    """获取或加载所选模型实例"""
    # 创建基于设备和所有选定模型名称的唯一缓存键
    model_config_key = f"{device}_{asr_model_name}_{vad_model_name}_{punc_model_name}_{spk_model_name}"

    if model_config_key not in model_cache:
        print(f"为配置 {model_config_key} 加载模型...")
        # 从字典中获取模型路径
        asr_path = asr_models.get(asr_model_name)
        vad_path = vad_models.get(vad_model_name)
        punc_path = punc_models.get(punc_model_name)
        spk_path = spk_models.get(spk_model_name)

        # 检查路径是否存在
        if not all([asr_path, vad_path, punc_path, spk_path]):
             # 哪个模型路径缺失的错误处理
             missing = []
             if not asr_path: missing.append(f"ASR: {asr_model_name}")
             if not vad_path: missing.append(f"VAD: {vad_model_name}")
             if not punc_path: missing.append(f"PUNC: {punc_model_name}")
             if not spk_path: missing.append(f"SPK: {spk_model_name}")
             raise ValueError(f"选定的一个或多个模型在 model.conf 中未找到路径: {', '.join(missing)}")

        model_params = {
            "disable_update": True,
            "device": device,
            "model": asr_path,
            "vad_model": vad_path,
            "punc_model": punc_path,
            "spk_model": spk_path,
            "timestamp": True,
        }
        try:
            model_cache[model_config_key] = AutoModel(**model_params)
            print(f"配置 {model_config_key} 的模型加载完成。")
        except Exception as e:
            print(f"加载模型时出错 ({model_config_key}): {e}")
            # 可以选择在这里清除缓存条目
            # del model_cache[model_config_key]
            raise  # 重新抛出异常，以便 Gradio 可以显示错误
    return model_cache[model_config_key]

# --- Gradio 的主要处理函数 ---
def process_audio(audio_file_path, device, asr_choice, vad_choice, punc_choice, spk_choice, use_srt_format, preset_spk_num):
    """处理上传的音频文件并根据选项返回结果文本"""
    if not audio_file_path:
        return "请先上传一个音频文件。"
    if not all([asr_choice, vad_choice, punc_choice, spk_choice]):
        return "错误：请确保所有模型（ASR, VAD, PUNC, SPK）都已选择。"

    # 处理说话人数量输入
    spk_num_arg = None
    if preset_spk_num is not None and int(preset_spk_num) > 0:
        spk_num_arg = int(preset_spk_num)
        print(f"指定说话人数量: {spk_num_arg}")
    else:
        print("未指定说话人数量，将自动检测。")

    print(f"接收到文件: {audio_file_path}, 设备: {device}, 模型: ASR={asr_choice}, VAD={vad_choice}, PUNC={punc_choice}, SPK={spk_choice}, SRT格式: {use_srt_format}")

    try:
        model = get_model(device, asr_choice, vad_choice, punc_choice, spk_choice)
        print(f"开始处理音频文件: {audio_file_path}")
        # 将 preset_spk_num 传递给 generate 函数
        res = model.generate(input=audio_file_path, preset_spk_num=spk_num_arg)
        print("识别完成，开始格式化结果...")

        formatted_results = []
        subtitle_index = 1 # SRT 格式需要序号

        for result in res:
            if 'sentence_info' in result:
                for sentence in result['sentence_info']:
                    spk_value = sentence.get('spk')
                    speaker = f"spk{spk_value + 1}" if spk_value is not None and isinstance(spk_value, int) else "spkunknown"
                    text = sentence.get('text', '').rstrip(',.。，!！?？')
                    start_time = format_timestamp(sentence.get('start', 0))
                    end_time = format_timestamp(sentence.get('end', 0))
                    
                    # 将说话人格式化为带方括号
                    speaker_formatted = f"[{speaker}]"
                    
                    if use_srt_format:
                        formatted_results.append(str(subtitle_index))
                        formatted_results.append(f"{start_time} --> {end_time}")
                        # 在 SRT 输出中为说话人添加方括号
                        formatted_results.append(f"{speaker_formatted}   {text}")
                        formatted_results.append("") # SRT 需要空行分隔
                        subtitle_index += 1
                    else:
                        # 在标准输出中为说话人添加方括号
                        formatted_line = f"{start_time} --> {end_time}  {speaker_formatted}   {text}"
                        formatted_results.append(formatted_line)

            elif 'timestamp' in result: # 备用逻辑
                 spk_value = result.get('spk')
                 speaker = f"spk{spk_value + 1}" if spk_value is not None and isinstance(spk_value, int) else "spkunknown"
                 text = result.get('text', '').rstrip(',.。，!！?？')
                 
                 # 将说话人格式化为带方括号
                 speaker_formatted = f"[{speaker}]"
                 
                 if result.get('timestamp'):
                     # 使用片段的第一个和最后一个时间戳
                     start_ms = result['timestamp'][0][0]
                     end_ms = result['timestamp'][-1][1]
                     start_time = format_timestamp(start_ms)
                     end_time = format_timestamp(end_ms)

                     if use_srt_format:
                         formatted_results.append(str(subtitle_index))
                         formatted_results.append(f"{start_time} --> {end_time}")
                         # 在 SRT 输出中为说话人添加方括号
                         formatted_results.append(f"{speaker_formatted}   {text}")
                         formatted_results.append("")
                         subtitle_index += 1
                     else:
                         # 在标准输出中为说话人添加方括号
                         formatted_line = f"{start_time} --> {end_time}  {speaker_formatted}   {text}"
                         formatted_results.append(formatted_line)
                 elif text: # 只有文本，没有时间戳
                     if use_srt_format:
                         # SRT 格式需要时间戳，跳过没有时间戳的片段
                         pass 
                     else:
                         # 在标准输出中为说话人添加方括号
                         formatted_results.append(f"{speaker_formatted}   {text}")
            elif result.get('text', '').strip(): # 只有文本，没有 sentence_info 或 timestamp 键
                text = result.get('text', '').rstrip(',.。，!！?？')
                # 没有时间戳无法生成 SRT
                if not use_srt_format:
                     # 此处说话人未知
                     formatted_results.append(text)

        output_text = "\n".join(formatted_results)
        print("结果格式化完成。")
        return output_text

    except Exception as e:
        print(f"处理音频时出错: {e}")
        import traceback
        traceback.print_exc()
        return f"处理失败: {e}"

# --- Gradio 界面定义 ---
with gr.Blocks() as demo:
    gr.Markdown("# FunASR 语音识别 (带时间戳和说话人)")
    gr.Markdown("上传音频文件，选择运行设备和模型进行识别。可选择输出 SRT 格式，并可指定说话人数量。")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="上传音频文件 (Upload Audio File)")
            device_select = gr.Radio(choices=["cuda", "cpu"], label="选择设备 (Select Device)", value="cuda")
            with gr.Row(): # 将复选框和数字输入放在一行
                preset_spk_num_input = gr.Number(label="指定说话人数 (0=自动)", value=0, minimum=0, step=1, scale=1) # 添加数字输入
                srt_format_checkbox = gr.Checkbox(label="输出 SRT 格式", value=False, scale=1)
                
            with gr.Accordion("模型配置 (Model Configuration)", open=False):
                asr_select = gr.Dropdown(choices=list(asr_models.keys()), value=default_asr, label="ASR 模型")
                vad_select = gr.Dropdown(choices=list(vad_models.keys()), value=default_vad, label="VAD 模型")
                punc_select = gr.Dropdown(choices=list(punc_models.keys()), value=default_punc, label="标点模型")
                spk_select = gr.Dropdown(choices=list(spk_models.keys()), value=default_spk, label="说话人模型")

            submit_button = gr.Button("开始识别 (Start Recognition)")

        with gr.Column(scale=2):
            text_output = gr.Textbox(label="识别结果 (Recognition Result)", lines=50, interactive=False)

    submit_button.click(
        fn=process_audio,
        # 输入列表添加说话人数量输入框
        inputs=[audio_input, device_select, asr_select, vad_select, punc_select, spk_select, srt_format_checkbox, preset_spk_num_input],
        outputs=[text_output]
    )

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7861)
