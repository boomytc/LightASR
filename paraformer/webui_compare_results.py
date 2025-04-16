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


# 生成模型平均CER或正确率对比柱状图
def plot_model_avg_cer(stats, model_names, metric_type='cer'):
    """生成模型平均CER或正确率对比柱状图"""
    if not stats or 'models' not in stats or not stats['models'] or not model_names:
        st.warning("没有足够的统计数据来生成对比图。")
        return None

    data = []
    for model_name in model_names:
        if model_name in stats['models']:
            value = stats['models'][model_name]['avg_cer']
            if metric_type == 'accuracy':
                value = 100 - value  # 正确率 = 100 - CER
            data.append({
                'model': model_name,
                'value': value
            })

    if not data:
        st.warning("在统计数据中找不到有效模型的数据。")
        return None

    # 根据指标类型排序
    if metric_type == 'cer':
        # 字错率降序排列（从高到低）
        data.sort(key=lambda x: x['value'], reverse=True)
    else:
        # 正确率升序排列（从低到高）
        data.sort(key=lambda x: x['value'])

    models = [item['model'] for item in data]
    values = [item['value'] for item in data]

    # 计算 Y 轴范围
    min_value = min(values) * 0.9 if min(values) > 0 else 0
    max_value = max(values) * 1.1

    # 设置颜色渐变
    if metric_type == 'cer':
        # 红色到绿色渐变(红色表示较高的错误率)
        colors = ['rgba(220,20,60,0.8)', 'rgba(255,69,0,0.8)', 'rgba(255,140,0,0.8)', 
                 'rgba(255,215,0,0.8)', 'rgba(154,205,50,0.8)', 'rgba(34,139,34,0.8)']
    else:
        # 绿色到红色渐变(绿色表示较高的正确率)
        colors = ['rgba(34,139,34,0.8)', 'rgba(154,205,50,0.8)', 'rgba(255,215,0,0.8)', 
                 'rgba(255,140,0,0.8)', 'rgba(255,69,0,0.8)', 'rgba(220,20,60,0.8)']

    # 确保颜色数量匹配模型数量
    model_colors = colors[:len(models)] if len(models) <= len(colors) else [colors[i % len(colors)] for i in range(len(models))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        text=[f"{val:.2f}%" for val in values],
        textposition='auto',
        marker_color=model_colors,
        marker_line_color='rgba(0,0,0,0.5)',
        marker_line_width=1,
        hoverinfo='x+y',
        hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
    ))

    title = '模型平均字错率(CER)对比' if metric_type == 'cer' else '模型平均正确率(Accuracy)对比'
    y_axis_title = '平均字错率 CER (%)' if metric_type == 'cer' else '平均正确率(Accuracy) (%)'  

    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='black')
        },
        xaxis_title={
            'text': '模型名称',
            'font': dict(size=16)
        },
        yaxis_title={
            'text': y_axis_title,
            'font': dict(size=16)
        },
        # 设置 Y 轴范围
        yaxis=dict(
            range=[min_value, max_value],
            gridcolor='lightgray'
        ),
        xaxis=dict(
            gridcolor='lightgray'
        ),
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,240,240,0.2)',
        bargap=0.3,
        margin=dict(l=50, r=50, t=80, b=80),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )

    return fig


# 生成模型平均CER或正确率对比折线图
def plot_model_avg_cer_line(stats, model_names, metric_type='cer'):
    """生成模型平均CER或正确率对比折线图"""
    if not stats or 'models' not in stats or not stats['models'] or not model_names:
        st.warning("没有足够的统计数据来生成折线图。")
        return None

    data = []
    for model_name in model_names:
        if model_name in stats['models']:
            value = stats['models'][model_name]['avg_cer']
            if metric_type == 'accuracy':
                value = 100 - value  # 正确率 = 100 - CER
            data.append({
                'model': model_name,
                'value': value
            })

    if not data:
        st.warning("在统计数据中找不到有效模型的数据。")
        return None

    # 根据指标类型排序
    if metric_type == 'cer':
        # 字错率降序排列（从高到低）
        data.sort(key=lambda x: x['value'], reverse=True)
    else:
        # 正确率升序排列（从低到高）
        data.sort(key=lambda x: x['value'])

    models = [item['model'] for item in data]
    values = [item['value'] for item in data]

    fig = go.Figure()

    name_text = '平均CER' if metric_type == 'cer' else '平均正确率'
    line_color = '#FF5733' if metric_type == 'cer' else '#33A1FF'
    
    fig.add_trace(go.Scatter(
        x=models,
        y=values,
        mode='lines+markers+text',
        name=name_text,
        text=[f"{val:.2f}%" for val in values],
        textposition="top center",
        textfont=dict(
            size=14,
            color='black'
        ),
        line=dict(
            color=line_color, 
            width=4,
            shape='spline',
            smoothing=1.3
        ),
        marker=dict(
            size=14,
            color=values,
            colorscale='Viridis' if metric_type == 'accuracy' else 'RdYlGn_r',
            line=dict(width=2, color='black')
        ),
        hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
    ))

    title = '模型平均字错率(CER)对比折线图' if metric_type == 'cer' else '模型平均正确率(Accuracy)对比折线图'
    y_axis_title = '平均字错率 CER (%)' if metric_type == 'cer' else '平均正确率(Accuracy) (%)'

    min_value = min(values) * 0.9 if min(values) > 0 else 0
    max_value = max(values) * 1.1

    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='black')
        },
        xaxis_title={
            'text': '模型名称',
            'font': dict(size=16)
        },
        yaxis_title={
            'text': y_axis_title,
            'font': dict(size=16)
        },
        height=600,
        template='plotly_white',
        plot_bgcolor='rgba(240,240,240,0.2)',
        margin=dict(l=50, r=50, t=80, b=80),
        yaxis=dict(
            range=[min_value, max_value],
            gridcolor='lightgray'
        ),
        xaxis=dict(
            gridcolor='lightgray'
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        showlegend=False
    )

    # 添加辅助线和区域
    fig.add_shape(
        type="line",
        x0=0,
        y0=sum(values)/len(values),
        x1=len(models)-1,
        y1=sum(values)/len(values),
        line=dict(
            color="gray",
            width=2,
            dash="dash",
        ),
        name="平均值"
    )
    
    # 添加平均值标签
    fig.add_annotation(
        x=len(models)-1,
        y=sum(values)/len(values),
        text=f"平均值: {sum(values)/len(values):.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray",
        font=dict(size=14)
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
        st.header("分析说明")
        st.markdown("""
        此工具用于可视化分析模型对比测试结果，包含以下内容：
        - 模型性能总览（字错率、处理时间）
        - 平均字错率/正确率对比图
        """)
        
        st.markdown("---")
        st.header("选择结果文件")

        uploaded_file = st.file_uploader(
            "上传结果文件进行分析",
            type=["txt"],
            key='result_uploader',
            help="上传文件后将自动开始分析。"
        )

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

                # 添加单选框选择显示字错率或正确率
                st.markdown("## 模型性能对比")
                metric_options = ["字错率(CER)", "正确率(Accuracy)"]
                selected_metric = st.radio("选择显示指标：", metric_options, horizontal=True)
                
                # 根据选择的指标类型显示相应的图表
                metric_type = 'cer' if selected_metric == "字错率(CER)" else 'accuracy'
                
                # 生成并显示柱状图
                avg_fig = plot_model_avg_cer(stats, model_names, metric_type)
                if avg_fig:
                    st.plotly_chart(avg_fig, use_container_width=True)
                else:
                    st.info(f"无法生成{selected_metric}柱状图。")
                
                # 生成并显示折线图
                avg_line_fig = plot_model_avg_cer_line(stats, model_names, metric_type)
                if avg_line_fig:
                    st.plotly_chart(avg_line_fig, use_container_width=True)
                else:
                    st.info(f"无法生成{selected_metric}折线图。")

    else:
        st.info("👈 请在侧边栏上传结果文件进行分析")
        st.markdown("## 模型性能总览")
        st.markdown("上传文件后将在此处显示模型性能统计表格和图表...")
        if st.session_state.parsed_data is not None:
            st.session_state.parsed_data = None
            st.session_state.loaded_content_hash = None


if __name__ == "__main__":
    main()