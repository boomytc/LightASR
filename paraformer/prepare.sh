#!/bin/bash

# 显示帮助信息的函数
show_help() {
    echo "用途: 生成用于语音识别训练的数据文件"
    echo
    echo "用法: $0 --data_dir <数据目录> --output_dir <输出目录>"
    echo "或使用: $0 -h 显示帮助信息"
    echo
    echo "参数说明:"
    echo "  --data_dir    包含音频和文本文件的目录"
    echo "  --output_dir  生成的文件的保存目录"
    echo
    echo "示例:"
    echo "  $0 --data_dir ./data --output_dir ./output"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$data_dir" ] || [ -z "$output_dir" ]; then
    echo "错误: 必须指定 --data_dir 和 --output_dir"
    show_help
fi

# 检查目录是否存在
if [ ! -d "$data_dir" ]; then
    echo "错误: 数据目录 '$data_dir' 不存在"
    exit 1
fi

# 检查是否安装了parallel
if ! command -v parallel &> /dev/null; then
    echo "请先安装 GNU parallel: sudo apt-get install parallel"
    exit 1
fi

# 设置并行任务数，使用可用CPU核心数
NUM_CORES=$(nproc)
export NUM_CORES

# 获取脚本目录
script_dir=$(dirname "$(realpath "$0")")

# 创建临时目录用于存储中间结果
tmp_dir="${output_dir}/tmp_$$"
mkdir -p "$tmp_dir"

# 创建输出文件
> "${output_dir}/train_wav.scp"
> "${output_dir}/train_text.txt"

# 创建统计文件
success_count_file="${tmp_dir}/success_count"
failed_count_file="${tmp_dir}/failed_count"
echo "0" > "$success_count_file"
echo "0" > "$failed_count_file"

# 确保统计文件存在且有正确的权限
chmod 666 "$success_count_file" "$failed_count_file"

# 获取当前时间戳作为前缀
timestamp=$(date +%Y%m%d_%H%M%S)

# 计算总文件数并导出为环境变量
total_files=$(find "$data_dir" -type f -name "*.txt" | wc -l)
export total_files
echo "找到 ${total_files} 个文本文件待处理..."

# 使用flock进行文件锁定的更新计数函数
update_count() {
    local count_file="$1"
    (
        flock -x 200
        if [ -f "$count_file" ]; then
            local current_count
            current_count=$(cat "$count_file")
            if [[ "$current_count" =~ ^[0-9]+$ ]]; then
                echo $((current_count + 1)) > "$count_file"
            fi
        fi
    ) 200>"${count_file}.lock"
}

# 定义处理单个文件的函数
process_file() {
    local txt_file="$1"
    local tmp_dir="$2"
    local timestamp="$3"
    local data_dir="$4"
    local script_dir="$5"
    local current_file=$PARALLEL_SEQ
    
    # 获取文件名(不含扩展名)
    filename=$(basename "$txt_file" .txt)
    
    # 创建临时计数文件
    local count_file="${tmp_dir}/${PARALLEL_SEQ}.count"
    echo "0" > "$count_file"
    
    # 首先获取所有匹配的文件列表
    matching_files=$(find "$data_dir" -type f \( \
        -name "${filename}.wav" -o \
        -name "${filename}.mp3" -o \
        -name "${filename}_*.wav" -o \
        -name "${filename}_*.mp3" \
    \))
    
    # 检查是否找到任何匹配文件
    if [ -z "$matching_files" ]; then
        echo "[$current_file/$total_files] 警告: '$txt_file' 没有找到匹配的音频文件" >&2
        update_count "$failed_count_file"
        return
    fi
    
    # 处理每个匹配的音频文件
    while read matching_audio; do
        if [ -f "$matching_audio" ]; then
            random_str=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
            audio_filename=$(basename "$matching_audio" | sed 's/\.[^.]*$//')
            unique_id="${timestamp}_${PARALLEL_SEQ}_${random_str}_${audio_filename}"
            
            # 处理文本内容，调用prepare.py中的process_text函数
            text=$(python3 -c "
import sys
sys.path.append('${script_dir}')
from prepare import process_text
print(process_text('${txt_file}'))
")
            
            if [ ! -z "$text" ]; then
                echo "$unique_id $(realpath "$matching_audio")" >> "${tmp_dir}/${PARALLEL_SEQ}.wav.scp"
                echo "$unique_id $text" >> "${tmp_dir}/${PARALLEL_SEQ}.text"
                curr_count=$(cat "$count_file")
                echo $((curr_count + 1)) > "$count_file"
                echo "[$current_file/$total_files] 成功匹配: '$txt_file' -> '$matching_audio'" >&2
            fi
        fi
    done <<< "$matching_files"
    
    # 获取最终的匹配计数
    matched_count=$(cat "$count_file")
    
    # 显示处理结果
    if [ "$matched_count" -eq 0 ]; then
        echo "[$current_file/$total_files] 警告: '$txt_file' 处理失败" >&2
        echo "原因可能是:" >&2
        echo "1. 没有找到匹配的音频文件" >&2
        echo "2. 文本内容处理后为空" >&2
        echo "3. 音频文件访问权限问题" >&2
        update_count "$failed_count_file"
    else
        echo "[$current_file/$total_files] 完成: '$txt_file' 共匹配到 $matched_count 个音频文件" >&2
        update_count "$success_count_file"
    fi
    
    # 清理临时计数文件
    rm -f "$count_file"
}
export -f process_file
export -f update_count
export success_count_file
export failed_count_file
export script_dir

# 并行处理所有txt文件，显示总进度条
echo "开始并行处理文件..."
find "$data_dir" -type f -name "*.txt" | \
    parallel --progress --bar --eta --jobs $NUM_CORES process_file {} "$tmp_dir" "$timestamp" "$data_dir" "$script_dir"

# 获取最终统计结果
success_count=$(cat "$success_count_file")
failed_count=$(cat "$failed_count_file")
total_pairs=$(cat "${tmp_dir}"/*.wav.scp 2>/dev/null | wc -l)

# 显示详细的处理统计
echo -e "\n处理统计："
echo "----------------------------------------"
echo "总文件数: ${total_files}"
echo "成功处理: ${success_count} 个文件"
echo "处理失败: ${failed_count} 个文件"
echo "生成音频-文本对: ${total_pairs} 对"
echo "----------------------------------------"

# 合并结果文件
echo "合并结果文件..."
cat "${tmp_dir}"/*.wav.scp > "${output_dir}/train_wav.scp"
cat "${tmp_dir}"/*.text > "${output_dir}/train_text.txt"

# 清理临时文件
rm -rf "$tmp_dir"

# 调用prepare.py生成jsonl文件
echo "开始生成jsonl文件..."
python3 "${script_dir}/prepare.py" --output_dir "${output_dir}"

if [ $? -eq 0 ]; then
    echo -e "\n所有文件生成完成!"
    echo "----------------------------------------"
    echo "处理结果统计："
    echo "- 总文本文件数：${total_files}"
    echo "- 成功处理文件：${success_count}"
    echo "- 处理失败文件：${failed_count}"
    echo "- 生成音频-文本对：${total_pairs}"
    echo "----------------------------------------"
    echo "输出文件："
    echo "- ${output_dir}/train_wav.scp"
    echo "- ${output_dir}/train_text.txt"
    echo "- ${output_dir}/train.jsonl"
    echo "- ${output_dir}/val.jsonl"
    echo "- ${output_dir}/test.jsonl"
else
    echo -e "\n错误: jsonl文件生成失败!"
    exit 1
fi
