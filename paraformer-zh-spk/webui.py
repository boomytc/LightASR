# coding=utf-8

import gradio as gr
import numpy as np
import torch
import torchaudio
import os
import configparser
from funasr import AutoModel

# 读取配置文件
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'model.conf'))

# 获取各种模型列表
asr_models = {}
for key in config['asr_models_dir']:
    asr_models[key] = config.get('asr_models_dir', key)

vad_models = {}
for key in config['vad_models_dir']:
    vad_models[key] = config.get('vad_models_dir', key)

punc_models = {}
for key in config['punc_models_dir']:
    punc_models[key] = config.get('punc_models_dir', key)

spk_models = {}
for key in config['spk_models_dir']:
    spk_models[key] = config.get('spk_models_dir', key)

# 默认模型配置
default_asr = list(asr_models.keys())[0] if asr_models else None
default_vad = list(vad_models.keys())[0] if vad_models else None
default_punc = list(punc_models.keys())[0] if punc_models else None
default_spk = list(spk_models.keys())[0] if spk_models else None

def format_timestamp(ms):
    """将毫秒转换为 00:00:00,000 格式"""
    # 处理None值
    if ms is None:
        ms = 0
    seconds = ms / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def load_model(asr_model, vad_model, punc_model, spk_model):
    """根据选择的模型路径加载模型"""
    return AutoModel(
        disable_update=True,
        model=asr_models[asr_model],
        vad_model=vad_models[vad_model],
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=punc_models[punc_model],
        spk_model=spk_models[spk_model],
    )

def model_inference(input_wav, language, mode="normal", distinguish_speaker=True, 
                   asr_model=default_asr, vad_model=default_vad, 
                   punc_model=default_punc, spk_model=default_spk, fs=16000):
    global model
    # 创建当前模型配置的唯一标识
    model_config_id = f"{asr_model}_{vad_model}_{punc_model}_{spk_model}"
    
    # 如果选择的模型与当前加载的不同，重新加载模型
    if not hasattr(model_inference, 'current_model') or model_inference.current_model != model_config_id:
        model = load_model(asr_model, vad_model, punc_model, spk_model)
        model_inference.current_model = model_config_id
    
    language_abbr = {"zh": "zh"}
    language = "zh"
    selected_language = language_abbr[language]
    
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    
    merge_vad = True
    print(f"language: {language}, merge_vad: {merge_vad}, mode: {mode}")
    
    if mode == "normal":
        text = model.generate(input=input_wav,
                            cache={},
                            language=language,
                            use_itn=True,
                            batch_size_s=60, 
                            merge_vad=merge_vad)
        return text[0]["text"]
    else:  # timestamp mode
        res = model.generate(
            input=input_wav,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=merge_vad,
            timestamp=True,
        )
        
        formatted_results = []
        subtitle_index = 1
        
        for result in res:
            if 'sentence_info' in result:
                for sentence in result['sentence_info']:
                    # 确保spk值存在
                    spk_value = sentence.get('spk')
                    speaker = f"[spk{spk_value if spk_value is not None else 'unknown'}]" if distinguish_speaker else ""
                    text = sentence.get('text', '').rstrip(',.。，!！?？') # 去除句尾标点
                    
                    # 确保start和end值存在
                    start_value = sentence.get('start')
                    end_value = sentence.get('end')
                    if start_value is None:
                        start_value = 0
                    if end_value is None:
                        end_value = 0
                        
                    start_time = format_timestamp(start_value)
                    end_time = format_timestamp(end_value)
                    
                    formatted_results.append(str(subtitle_index))
                    formatted_results.append(f"{start_time} --> {end_time}")
                    formatted_results.append(f"{speaker} {text}")
                    formatted_results.append("")
                    subtitle_index += 1
                    
            elif 'timestamp' in result:
                # 确保spk值存在
                spk_value = result.get('spk')
                speaker = f"[spk{spk_value if spk_value is not None else 'unknown'}]" if distinguish_speaker else ""
                text = result.get('text', '').rstrip(',.。，!！?？') # 去除句尾标点
                
                for ts in result['timestamp']:
                    # 确保时间戳值存在
                    start_value = ts[0] if len(ts) > 0 and ts[0] is not None else 0
                    end_value = ts[1] if len(ts) > 1 and ts[1] is not None else 0
                    
                    start_time = format_timestamp(start_value)
                    end_time = format_timestamp(end_value)
                    
                    formatted_results.append(str(subtitle_index))
                    formatted_results.append(f"{start_time} --> {end_time}")
                    formatted_results.append(f"{speaker} {text}")
                    formatted_results.append("")
                    subtitle_index += 1
            else:
                formatted_results.append(str(subtitle_index))
                formatted_results.append("00:00:00,000 --> 00:00:00,000")
                formatted_results.append(result.get('text', ''))
                formatted_results.append("")
                subtitle_index += 1
        
        return "\n".join(formatted_results)

html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">使用说明</h2>
    <p style="font-size: 18px;margin-left: 20px;">上传音频文件或通过麦克风输入，然后配置语言和识别模式。音频将被转录为相应的文本。</p>
</div>
"""


def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column(scale=1):
                audio_inputs = gr.Audio(label="上传音频或使用麦克风")
                
                with gr.Accordion("配置"):
                    with gr.Row():
                        language_inputs = gr.Dropdown(
                            choices=["zh"],
                            value="zh",
                            label="语言",
                            scale=1
                        )
                        mode_inputs = gr.Dropdown(
                            choices=["normal", "timestamp"],
                            value="normal",
                            label="识别模式",
                            scale=1
                        )
                        with gr.Column():
                            distinguish_speaker_inputs = gr.Radio(
                                choices=["是", "否"],
                                value="否",
                                label="说话人模式",
                                scale=1
                            )
                    
                    # 添加模型选择下拉框
                    with gr.Row():
                        asr_model_inputs = gr.Dropdown(
                            choices=list(asr_models.keys()),
                            value=default_asr,
                            label="ASR模型选择",
                            scale=1
                        )
                    
                    with gr.Row():
                        vad_model_inputs = gr.Dropdown(
                            choices=list(vad_models.keys()),
                            value=default_vad,
                            label="VAD模型选择",
                            scale=1
                        )
                    
                    with gr.Row():
                        punc_model_inputs = gr.Dropdown(
                            choices=list(punc_models.keys()),
                            value=default_punc,
                            label="标点模型选择",
                            scale=1
                        )
                    
                    with gr.Row():
                        spk_model_inputs = gr.Dropdown(
                            choices=list(spk_models.keys()),
                            value=default_spk,
                            label="说话人模型选择",
                            scale=1
                        )
                
                fn_button = gr.Button("开始识别", variant="primary")
                
            # 添加第二列用于展示识别结果
            with gr.Column(scale=1):
                text_outputs = gr.Textbox(label="识别结果", lines=50, max_lines=50)
        
        fn_button.click(
            lambda *args: model_inference(
                args[0],  # audio_inputs
                args[1],  # language_inputs
                args[2],  # mode_inputs
                args[3] == "是",  # distinguish_speaker_inputs
                args[4],  # asr_model_inputs
                args[5],  # vad_model_inputs
                args[6],  # punc_model_inputs
                args[7]   # spk_model_inputs
            ),
            inputs=[
                audio_inputs, 
                language_inputs, 
                mode_inputs, 
                distinguish_speaker_inputs, 
                asr_model_inputs,
                vad_model_inputs,
                punc_model_inputs,
                spk_model_inputs
            ], 
            outputs=text_outputs
        )

    demo.launch()


if __name__ == "__main__":
    # iface.launch()
    launch()
