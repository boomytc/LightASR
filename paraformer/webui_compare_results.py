#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
import re
import matplotlib as mpl
import plotly.graph_objects as go
import hashlib

# 设置页面配置
st.set_page_config(
    page_title="语音识别模型对比分析",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False


# 修改后的解析函数，可以接受文件对象或文件路径
def parse_results_file(file_input):
    """
    解析模型对比结果文件 (可接受文件路径或文件对象)
    返回：
    1. 模型名称列表 (基于摘要统计)
    2. 每个文件的详细结果字典
    3. 总体统计信息字典
    """
    content = None
    file_source_info = "未知来源"  # For error messages

    # Check if input is a Streamlit UploadedFile object
    if hasattr(file_input, 'getvalue') and hasattr(file_input, 'name'):
        file_source_info = f"上传的文件 '{file_input.name}'"
        try:
            content_bytes = file_input.getvalue()
            content = content_bytes.decode('utf-8')
        except Exception as e:
            st.error(f"读取 {file_source_info} 内容失败: {e}")
            return None, None, None, None
    elif isinstance(file_input, str):
        file_source_info = f"文件路径 '{file_input}'"
        if not os.path.exists(file_input):
            st.error(f"结果文件不存在：{file_input}")
            return None, None, None, None
        try:
            with open(file_input, 'r', encoding='utf-8') as f:
                content = f.read()
                with open(file_input, 'rb') as fb:
                    content_bytes = fb.read()
        except Exception as e:
            st.error(f"读取 {file_source_info} 失败: {e}")
            return None, None, None, None
    else:
        st.error("无效的文件输入类型")
        return None, None, None, None

    if content is None:
        st.error(f"无法从 {file_source_info} 获取内容。")
        return None, None, None, None

    content_hash = hashlib.md5(content_bytes).hexdigest()

    parts = content.split("=== 测试结果统计 ===")
    if len(parts) < 2:
        st.error(f"{file_source_info} 格式不正确，缺少统计摘要部分")
        return None, None, None, content_hash

    detail_part = parts[0]
    summary_part = parts[1]

    file_results = []
    model_names_from_detail = set()

    file_pattern = r'\n\[\d+/\d+\] 文件: (.+)\n参考文本: (.+)\n参考文本\(无标点\): (.+)\n(.*?)-{80}'
    file_blocks = re.findall(file_pattern, detail_part, re.DOTALL)

    for file_block in file_blocks:
        file_name, ref_text, ref_text_no_punc, model_results_text = file_block
        file_result = {
            'file_name': file_name,
            'reference_text': ref_text,
            'reference_text_no_punc': ref_text_no_punc,
            'models': {}
        }

        model_pattern = r'(\S+) 识别结果: (.+)\n\1 CER: ([\d.]+)%'
        model_matches = re.findall(model_pattern, model_results_text)

        for model_match in model_matches:
            model_name, rec_text, cer = model_match
            model_names_from_detail.add(model_name)
            file_result['models'][model_name] = {
                'text': rec_text,
                'cer': float(cer)
            }

        file_results.append(file_result)

    stats = {}
    total_files_match = re.search(r'总测试数据数量: (\d+)', summary_part)
    total_time_match = re.search(r'整体任务总耗时: ([\d.]+) 秒', summary_part)

    if total_files_match and total_time_match:
        stats['total_files'] = int(total_files_match.group(1))
        stats['total_time'] = float(total_time_match.group(1))

    model_pattern = r'(\S+):\n成功处理数量: (\d+)/\d+\n平均字错率\(CER\): ([\d.]+)%\n平均识别耗时\(单条\): ([\d.]+)秒'
    model_matches = re.findall(model_pattern, summary_part)

    stats['models'] = {}
    model_names_from_summary = []
    for model_match in model_matches:
        model_name, success_count, avg_cer, avg_time = model_match
        model_name = model_name.strip(':')
        stats['models'][model_name] = {
            'success_count': int(success_count),
            'avg_cer': float(avg_cer),
            'avg_time': float(avg_time)
        }
        if model_name not in model_names_from_summary:
            model_names_from_summary.append(model_name)

    if not stats.get('models'):
        st.error(f"未能从 {file_source_info} 的摘要部分解析出任何模型统计信息。")
        if model_names_from_detail:
            st.warning("将尝试使用从文件详情中解析出的模型名称。")
            return list(model_names_from_detail), file_results, stats, content_hash
        else:
            return None, None, None, content_hash

    return model_names_from_summary, file_results, stats, content_hash


# 生成汇总统计表格
def generate_summary_table(stats):
    """生成模型性能汇总统计表格"""
    if not stats or 'models' not in stats:
        return None

    data = []
    for model_name, model_stats in stats['models'].items():
        data.append({
            '模型名称': model_name,
            '成功识别数量': model_stats['success_count'],
            '平均字错率(CER)': f"{model_stats['avg_cer']:.2f}%",
            '平均识别耗时(秒/条)': f"{model_stats['avg_time']:.3f}"
        })

    return pd.DataFrame(data)


# 生成模型平均CER对比柱状图
def plot_model_avg_cer(stats, model_names):
    """生成模型平均CER对比柱状图"""
    if not stats or 'models' not in stats or not stats['models'] or not model_names:
        st.warning("没有足够的统计数据来生成平均CER对比图。")
        return None

    models = []
    avg_cers = []

    for model_name in model_names:
        if model_name in stats['models']:
            models.append(model_name)
            avg_cers.append(stats['models'][model_name]['avg_cer'])

    if not models:
        st.warning("在统计数据中找不到有效模型的平均CER数据。")
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=avg_cers,
        text=[f"{cer:.2f}%" for cer in avg_cers],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(models)]
    ))

    fig.update_layout(
        title='模型平均字错率(CER)对比',
        xaxis_title='模型名称',
        yaxis_title='平均字错率 CER (%)',
        height=500
    )

    return fig


# 生成模型平均CER对比折线图
def plot_model_avg_cer_line(stats, model_names):
    """生成模型平均CER对比折线图"""
    if not stats or 'models' not in stats or not stats['models'] or not model_names:
        st.warning("没有足够的统计数据来生成平均CER折线图。")
        return None

    models = []
    avg_cers = []

    for model_name in model_names:
        if model_name in stats['models']:
            models.append(model_name)
            avg_cers.append(stats['models'][model_name]['avg_cer'])

    if not models:
        st.warning("在统计数据中找不到有效模型的平均CER数据。")
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=models,
        y=avg_cers,
        mode='lines+markers+text',
        name='平均CER',
        text=[f"{cer:.2f}%" for cer in avg_cers],
        textposition="top center",
        line=dict(color='royalblue', width=3),
        marker=dict(size=12)
    ))

    fig.update_layout(
        title='模型平均字错率(CER)对比折线图',
        xaxis_title='模型名称',
        yaxis_title='平均字错率 CER (%)',
        height=500,
        yaxis=dict(
            range=[0, max(avg_cers) * 1.2]
        )
    )

    return fig


# 主函数
def main():
    st.title("🎤 语音识别模型对比分析")

    if 'loaded_content_hash' not in st.session_state:
        st.session_state['loaded_content_hash'] = None
    if 'parsed_data' not in st.session_state:
        st.session_state['parsed_data'] = None

    with st.sidebar:
        st.header("选择结果文件")

        uploaded_file = st.file_uploader(
            "上传结果文件进行分析",
            type=["txt"],
            key='result_uploader',
            help="上传文件后将自动开始分析。"
        )

        st.markdown("---")
        st.markdown("### 分析说明")
        st.markdown("""
        此工具用于可视化分析模型对比测试结果，包含以下内容：
        - 模型性能总览（字错率、处理时间）
        - 平均字错率对比图
        """)

    if uploaded_file is not None:
        model_names, file_results, stats, current_content_hash = parse_results_file(uploaded_file)

        if current_content_hash is not None and \
           (st.session_state.loaded_content_hash != current_content_hash or st.session_state.parsed_data is None):

            st.session_state.loaded_content_hash = current_content_hash

            if model_names is not None and file_results is not None and stats is not None:
                st.session_state.parsed_data = (model_names, file_results, stats)
            else:
                st.session_state.parsed_data = None
                st.warning("文件解析失败，请检查文件内容和格式。上方可能显示具体错误信息。")

        if st.session_state.parsed_data is not None:
            if st.session_state.loaded_content_hash == current_content_hash:
                model_names, file_results, stats = st.session_state.parsed_data

                if not model_names:
                    st.warning("未从结果文件的摘要中解析出任何模型名称。")
                if not stats or 'models' not in stats or not stats['models']:
                    st.warning("未从结果文件中解析出有效的统计摘要。")

                total_files = stats.get('total_files', 0)
                total_time = stats.get('total_time', 0)

                st.markdown(f"### 文件: `{uploaded_file.name}`")
                st.markdown(f"## 基本信息")
                col1, col2, col3 = st.columns(3)
                col1.metric("总测试文件数", f"{total_files}")
                col2.metric("总测试时长", f"{total_time:.2f}秒")
                col3.metric("模型数量", f"{len(model_names)}")

                st.markdown("## 模型性能总览")
                summary_table = generate_summary_table(stats)
                if summary_table is not None and not summary_table.empty:
                    st.dataframe(summary_table, use_container_width=True)
                else:
                    st.info("没有足够的统计数据来显示模型性能总览。")

                st.markdown("## 模型平均字错率(CER)对比")
                avg_cer_fig = plot_model_avg_cer(stats, model_names)
                if avg_cer_fig:
                    st.plotly_chart(avg_cer_fig, use_container_width=True)
                else:
                    st.info("无法生成平均CER柱状图。")

                avg_cer_line_fig = plot_model_avg_cer_line(stats, model_names)
                if avg_cer_line_fig:
                    st.plotly_chart(avg_cer_line_fig, use_container_width=True)
                else:
                    st.info("无法生成平均CER折线图。")

    else:
        st.info("👈 请在侧边栏上传结果文件进行分析")
        st.markdown("## 模型性能总览")
        st.markdown("上传文件后将在此处显示模型性能统计表格和图表...")
        if st.session_state.parsed_data is not None:
            st.session_state.parsed_data = None
            st.session_state.loaded_content_hash = None


if __name__ == "__main__":
    main()