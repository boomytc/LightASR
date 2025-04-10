#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
import re
import matplotlib as mpl
import plotly.graph_objects as go

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


# 解析 results.txt 文件
def parse_results_file(file_path):
    """
    解析模型对比结果文件
    返回：
    1. 模型名称列表 (基于摘要统计)
    2. 每个文件的详细结果字典
    3. 总体统计信息字典
    """
    if not os.path.exists(file_path):
        st.error(f"结果文件不存在：{file_path}")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割成两部分：详细结果和统计摘要
    parts = content.split("=== 测试结果统计 ===")
    
    if len(parts) < 2:
        st.error("结果文件格式不正确，缺少统计摘要部分")
        return None, None, None
    
    detail_part = parts[0]
    summary_part = parts[1]
    
    # 解析详细结果 (仍然解析，用于 file_results)
    file_results = []
    model_names_from_detail = set() # 用于从详情中收集名称，但不作为主要返回
    
    # 使用正则表达式匹配文件块
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
        
        # 解析每个模型的结果
        model_pattern = r'(\S+) 识别结果: (.+)\n\1 CER: ([\d.]+)%'
        model_matches = re.findall(model_pattern, model_results_text)
        
        for model_match in model_matches:
            model_name, rec_text, cer = model_match
            model_names_from_detail.add(model_name) # 收集详情中的模型名
            file_result['models'][model_name] = {
                'text': rec_text,
                'cer': float(cer)
            }
        
        file_results.append(file_result)
    
    # 解析统计摘要
    stats = {}
    total_files_match = re.search(r'总测试数据数量: (\d+)', summary_part)
    total_time_match = re.search(r'整体任务总耗时: ([\d.]+) 秒', summary_part)
    
    if total_files_match and total_time_match:
        stats['total_files'] = int(total_files_match.group(1))
        stats['total_time'] = float(total_time_match.group(1))
    
    # 解析每个模型的统计信息
    model_pattern = r'(\S+):\n成功处理数量: (\d+)/\d+\n平均字错率\(CER\): ([\d.]+)%\n平均识别耗时\(单条\): ([\d.]+)秒'
    model_matches = re.findall(model_pattern, summary_part)
    
    stats['models'] = {}
    model_names_from_summary = [] # 用于从摘要中收集名称
    for model_match in model_matches:
        model_name, success_count, avg_cer, avg_time = model_match
        # 确保证模型名称不包含冒号等结尾字符
        model_name = model_name.strip(':') 
        stats['models'][model_name] = {
            'success_count': int(success_count),
            'avg_cer': float(avg_cer),
            'avg_time': float(avg_time)
        }
        if model_name not in model_names_from_summary:
             model_names_from_summary.append(model_name)

    # 检查 stats['models'] 是否为空
    if not stats.get('models'):
        st.error("未能从结果文件的摘要部分解析出任何模型统计信息。")
        # 如果摘要为空，尝试使用从详情解析出的名称，虽然可能不一致
        if model_names_from_detail:
             st.warning("将尝试使用从文件详情中解析出的模型名称。")
             return list(model_names_from_detail), file_results, stats
        else:
             return None, None, None # 如果两部分都解析失败

    # 返回基于摘要统计的模型名称列表
    return model_names_from_summary, file_results, stats


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
    
    # 提取数据
    models = []
    avg_cers = []
    
    for model_name in model_names:
        if model_name in stats['models']:
            models.append(model_name)
            avg_cers.append(stats['models'][model_name]['avg_cer'])
    
    if not models:  # 检查是否有有效模型数据
        st.warning("在统计数据中找不到有效模型的平均CER数据。")
        return None
    
    # 创建条形图
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
    
    # 提取数据
    models = []
    avg_cers = []
    
    for model_name in model_names:
        if model_name in stats['models']:
            models.append(model_name)
            avg_cers.append(stats['models'][model_name]['avg_cer'])
    
    if not models:  # 检查是否有有效模型数据
        st.warning("在统计数据中找不到有效模型的平均CER数据。")
        return None
    
    # 创建折线图
    fig = go.Figure()
    
    # 添加折线
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
    
    # 设置图表布局
    fig.update_layout(
        title='模型平均字错率(CER)对比折线图',
        xaxis_title='模型名称',
        yaxis_title='平均字错率 CER (%)',
        height=500,
        yaxis=dict(
            # 设置y轴从0开始，使视觉对比更直观
            range=[0, max(avg_cers) * 1.2]  # 上限为最大值的1.2倍，留出文本空间
        )
    )
    
    return fig


# 主函数
def main():
    st.title("🎤 语音识别模型对比分析")
    
    # 侧边栏：文件选择
    with st.sidebar:
        st.header("选择结果文件")
        result_path = st.text_input(
            "结果文件路径", 
            placeholder="例如：/path/to/results.txt"
        )
        
        load_btn = st.button("加载数据")
        
        st.markdown("---")
        st.markdown("### 分析说明")
        st.markdown("""
        此工具用于可视化分析模型对比测试结果，包含以下内容：
        - 模型性能总览（字错率、处理时间）
        - 平均字错率对比图
        """)
    
    # 主界面
    if not result_path:
        st.info("👈 请在侧边栏输入结果文件路径并点击加载数据")
        
        # 显示示例UI
        st.markdown("## 模型性能总览")
        st.markdown("加载数据后将显示模型性能统计表格和图表...")
        
        return
    
    # 如果点击了加载按钮，解析结果文件
    if load_btn or 'model_names' in st.session_state:
        with st.spinner("正在解析结果文件..."):
            if load_btn or 'file_results' not in st.session_state:
                model_names, file_results, stats = parse_results_file(result_path)
                
                if model_names is None:
                    st.error("解析结果文件失败，请检查文件路径和格式")
                    return
                
                # 将结果存储在session_state中
                st.session_state.model_names = model_names
                st.session_state.file_results = file_results
                st.session_state.stats = stats
            else:
                model_names = st.session_state.model_names
                file_results = st.session_state.file_results
                stats = st.session_state.stats
        
        # 确保数据已加载
        if 'model_names' not in st.session_state or \
           'file_results' not in st.session_state or \
           'stats' not in st.session_state:
            st.error("无法加载或解析数据，请重试。")
            return

        model_names = st.session_state.model_names
        stats = st.session_state.stats

        # 增加对加载数据的检查
        if not model_names:
            st.warning("未从结果文件的摘要中解析出任何模型名称。") 
        if not stats or 'models' not in stats or not stats['models']:
            st.warning("未从结果文件中解析出有效的统计摘要。")

        # 显示基本信息
        total_files = stats.get('total_files', 0)
        total_time = stats.get('total_time', 0)
        
        st.markdown(f"## 基本信息")
        col1, col2, col3 = st.columns(3)
        col1.metric("总测试文件数", f"{total_files}")
        col2.metric("总测试时长", f"{total_time:.2f}秒")
        col3.metric("模型数量", f"{len(model_names)}")
        
        # 模型性能总览
        st.markdown("## 模型性能总览")
        summary_table = generate_summary_table(stats)
        if summary_table is not None and not summary_table.empty:
            st.dataframe(summary_table, use_container_width=True)
        else:
            st.info("没有足够的统计数据来显示模型性能总览。")
        
        # 平均CER对比图表
        st.markdown("## 模型平均字错率(CER)对比")
        avg_cer_fig = plot_model_avg_cer(stats, model_names)
        if avg_cer_fig:
            st.plotly_chart(avg_cer_fig, use_container_width=True)
        else:
            st.info("无法生成平均CER柱状图。")
        
        # 添加CER对比折线图
        avg_cer_line_fig = plot_model_avg_cer_line(stats, model_names)
        if avg_cer_line_fig:
            st.plotly_chart(avg_cer_line_fig, use_container_width=True)
        else:
            st.info("无法生成平均CER折线图。")


if __name__ == "__main__":
    main()