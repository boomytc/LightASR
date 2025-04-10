from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, QRadioButton, 
                           QButtonGroup, QSpinBox, QWidget, QMessageBox, QInputDialog, 
                           QListWidget, QPlainTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QProcess, pyqtSlot
import subprocess
import sys
import os

class CompareUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型对比工具")
        self.setGeometry(100, 100, 1000, 800)

        self.models = []
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # 创建分割器，上面是控制面板，下面是终端输出
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 控制面板部分
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)

        # 模型选择部分
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        model_button = QPushButton("添加模型")
        model_button.clicked.connect(self.add_model)
        remove_model_button = QPushButton("删除模型")  # 新增删除模型按钮
        remove_model_button.clicked.connect(self.remove_model)  # 绑定删除模型方法
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_button)
        model_layout.addWidget(remove_model_button)  # 添加删除按钮到布局
        layout.addLayout(model_layout)

        # 模型列表
        self.model_list = QListWidget()
        # 设置模型列表的固定高度，大约五行高度
        self.model_list.setFixedHeight(120)
        layout.addWidget(self.model_list)

        # 数据类型选择部分
        self.data_type_group = QButtonGroup()
        jsonl_radio = QRadioButton("JSONL")
        wav_txt_radio = QRadioButton("WAV/TXT")
        self.data_type_group.addButton(jsonl_radio)
        self.data_type_group.addButton(wav_txt_radio)
        jsonl_radio.setChecked(True)
        layout.addWidget(jsonl_radio)
        layout.addWidget(wav_txt_radio)

        # JSONL 数据选择
        self.jsonl_path = QLineEdit()
        self.jsonl_button = QPushButton("选择 JSONL 文件")  # 使用实例变量保存按钮引用
        self.jsonl_button.setObjectName("选择_JSONL_文件")  # 设置对象名称
        self.jsonl_button.clicked.connect(self.select_jsonl)
        jsonl_layout = QHBoxLayout()
        jsonl_layout.addWidget(self.jsonl_path)
        jsonl_layout.addWidget(self.jsonl_button)
        layout.addLayout(jsonl_layout)

        # WAV/TXT 数据选择
        self.wav_path = QLineEdit()
        self.txt_path = QLineEdit()
        self.wav_button = QPushButton("选择 WAV 目录")  # 使用实例变量保存按钮引用
        self.wav_button.setObjectName("选择_WAV_目录")  # 设置对象名称
        self.txt_button = QPushButton("选择 TXT 目录")  # 使用实例变量保存按钮引用
        self.txt_button.setObjectName("选择_TXT_目录")  # 设置对象名称
        self.wav_button.clicked.connect(self.select_wav)
        self.txt_button.clicked.connect(self.select_txt)
        wav_layout = QHBoxLayout()
        wav_layout.addWidget(self.wav_path)
        wav_layout.addWidget(self.wav_button)
        txt_layout = QHBoxLayout()
        txt_layout.addWidget(self.txt_path)
        txt_layout.addWidget(self.txt_button)
        layout.addLayout(wav_layout)
        layout.addLayout(txt_layout)

        # 设置初始状态
        self.update_data_type_inputs()

        # 绑定单选框状态变化事件
        jsonl_radio.toggled.connect(self.update_data_type_inputs)
        wav_txt_radio.toggled.connect(self.update_data_type_inputs)

        # 结果保存路径选择
        self.result_path = QLineEdit()
        result_button = QPushButton("选择保存路径")
        result_button.clicked.connect(self.select_result)
        open_result_button = QPushButton("打开路径")
        open_result_button.clicked.connect(self.open_result_path)
        result_layout = QHBoxLayout()
        result_layout.addWidget(self.result_path)
        result_layout.addWidget(result_button)
        result_layout.addWidget(open_result_button)
        layout.addLayout(result_layout)

        # 结果文件名设置 (新增)
        filename_layout = QHBoxLayout()
        filename_label = QLabel("结果文件名 (可选):")
        self.result_filename = QLineEdit()
        self.result_filename.setPlaceholderText("默认: results.txt") # 提示默认值
        filename_layout.addWidget(filename_label)
        filename_layout.addWidget(self.result_filename)
        layout.addLayout(filename_layout)

        # 每 GPU 进程数设置
        proc_layout = QHBoxLayout()
        proc_label = QLabel("每 GPU 进程数:")
        self.proc_per_gpu = QSpinBox()
        self.proc_per_gpu.setRange(1, 10)
        self.proc_per_gpu.setValue(1)
        proc_layout.addWidget(proc_label)
        proc_layout.addWidget(self.proc_per_gpu)
        layout.addLayout(proc_layout)

        # 确认按钮 和 结果分析按钮 (修改)
        button_layout = QHBoxLayout() # 使用水平布局放置按钮
        confirm_button = QPushButton("确认")
        confirm_button.clicked.connect(self.confirm)
        analyze_button = QPushButton("结果分析") # 新增结果分析按钮
        analyze_button.clicked.connect(self.analyze_results) # 绑定分析方法
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(analyze_button)
        layout.addLayout(button_layout) # 将按钮布局添加到主布局
        
        control_panel.setLayout(layout)
        
        # 终端输出部分
        terminal_widget = QWidget()
        terminal_layout = QVBoxLayout(terminal_widget)
        
        # 添加终端标签
        terminal_label = QLabel("终端输出:")
        terminal_layout.addWidget(terminal_label)
        
        # 添加终端输出文本框
        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.terminal_output.setStyleSheet("background-color: black; color: white; font-family: Consolas, Courier New, monospace;")
        terminal_layout.addWidget(self.terminal_output)
        
        # 添加控制面板和终端到分割器
        splitter.addWidget(control_panel)
        splitter.addWidget(terminal_widget)
        splitter.setSizes([350, 450])  # 调整初始大小比例，让终端窗口更高
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

    def add_model(self):
        model_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if model_path:
            model_name, ok = QInputDialog.getText(self, "输入模型名称", "请输入模型名称:")
            if ok and model_name:
                self.models.append((model_name, model_path))
                # 在列表中显示模型名称和路径
                self.model_list.addItem(f"名称: {model_name}, 路径: {model_path}")

    def remove_model(self):
        """从列表中删除选定的模型"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的模型！")
            return
            
        for item in selected_items:
            row = self.model_list.row(item)
            if 0 <= row < len(self.models):
                model_name = self.models[row][0]
                # 询问用户确认删除
                reply = QMessageBox.question(self, "确认删除", 
                                            f"确定要删除模型 '{model_name}' 吗？", 
                                            QMessageBox.StandardButton.Yes | 
                                            QMessageBox.StandardButton.No, 
                                            QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    # 从models列表和列表控件中删除模型
                    self.models.pop(row)
                    self.model_list.takeItem(row)

    def select_jsonl(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 JSONL 文件", "", "JSONL Files (*.jsonl)")
        if path:
            self.jsonl_path.setText(path)

    def select_wav(self):
        path = QFileDialog.getExistingDirectory(self, "选择 WAV 目录")
        if path:
            self.wav_path.setText(path)

    def select_txt(self):
        path = QFileDialog.getExistingDirectory(self, "选择 TXT 目录")
        if path:
            self.txt_path.setText(path)

    def select_result(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if path:
            self.result_path.setText(path)
            
    def open_result_path(self):
        path = self.result_path.text()
        if not path:
            QMessageBox.warning(self, "警告", "路径为空，请先选择保存路径！")
            return
            
        try:
            # 使用系统默认的文件管理器打开路径
            if sys.platform == "win32":
                subprocess.Popen(["explorer", path])
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开路径: {e}")

    def update_data_type_inputs(self):
        """根据选择的数据类型更新输入框和按钮的可用状态"""
        is_jsonl = self.data_type_group.checkedButton().text() == "JSONL"
        
        # JSONL 输入框和按钮
        self.jsonl_path.setEnabled(is_jsonl)
        self.jsonl_path.setStyleSheet("" if is_jsonl else "background-color: lightgray;")
        self.jsonl_button.setEnabled(is_jsonl)  # 使用实例变量直接引用按钮
        
        # WAV/TXT 输入框和按钮
        for widget in [self.wav_path, self.txt_path]:
            widget.setEnabled(not is_jsonl)
            widget.setStyleSheet("" if not is_jsonl else "background-color: lightgray;")
        self.wav_button.setEnabled(not is_jsonl)  # 使用实例变量直接引用按钮
        self.txt_button.setEnabled(not is_jsonl)  # 使用实例变量直接引用按钮

    def confirm(self):
        models_str = []
        for name, path in self.models:
            models_str.append(name)
            models_str.append(path)
        data_type = "jsonl" if self.data_type_group.checkedButton().text() == "JSONL" else "wav_txt"
        
        # 获取自定义文件名，如果为空则使用默认值
        output_filename = self.result_filename.text().strip()
        if not output_filename:
            output_filename = "results.txt"
        # 确保文件名以 .txt 结尾
        if not output_filename.endswith(".txt"):
            output_filename += ".txt"
            
        result = {
            "models": models_str,
            "data_type": data_type,
            "jsonl_path": self.jsonl_path.text(),
            "wav_path": self.wav_path.text(),
            "txt_path": self.txt_path.text(),
            "result_path": self.result_path.text(),
            "proc_per_gpu": self.proc_per_gpu.value(),
            "output_filename": output_filename # 添加文件名到字典
        }

        # 验证保存路径是否为空
        if not result["result_path"]:
            QMessageBox.warning(self, "警告", "请先选择结果保存路径！")
            return
        
        # 验证模型列表是否为空
        if not result["models"]:
            QMessageBox.warning(self, "警告", "请至少添加一个模型！")
            return

        # 清空终端输出
        self.terminal_output.clear()
        self.terminal_output.appendPlainText("开始执行对比程序...") 

        # 构造命令行参数
        command = [
            "python", "-u", "/media/fl01/data02/WorkSpace/LightASR/paraformer/compare_finetuned_unified.py", # 注意这里的路径可能需要根据实际情况调整
            "--models", *result["models"],
            "--result", result["result_path"],
            "--proc_per_gpu", str(result["proc_per_gpu"]),
            "--output_filename", result["output_filename"] # 添加自定义文件名参数
        ]

        if data_type == "jsonl":
            # 验证 JSONL 路径是否为空
            if not result["jsonl_path"]:
                QMessageBox.warning(self, "警告", "请选择 JSONL 文件路径！")
                return
            command.extend(["--jsonl", result["jsonl_path"]])
        else:
            # 验证 WAV/TXT 路径是否为空
            if not result["wav_path"] or not result["txt_path"]:
                QMessageBox.warning(self, "警告", "请选择 WAV 和 TXT 目录路径！")
                return
            command.extend(["--wav_txt_dirs", result["wav_path"], result["txt_path"]])

        # 在终端中显示将要执行的命令
        self.terminal_output.appendPlainText("执行命令: " + " ".join(command))
        self.terminal_output.appendPlainText("-----------------------------------")

        # 使用QProcess异步运行命令
        try:
            # 设置工作目录为脚本所在目录的上层目录，以便脚本能正确找到相对路径
            script_dir = os.path.dirname("/media/fl01/data02/WorkSpace/LightASR/paraformer/compare_finetuned_unified.py")
            self.process.setWorkingDirectory(script_dir)
            self.process.start(command[0], command[1:])
        except Exception as e:
            self.terminal_output.appendHtml(f'<span style="color:red;">启动进程失败: {e}</span>')
            QMessageBox.critical(self, "错误", f"启动进程失败: {e}")

    def analyze_results(self):
        """执行 streamlit run webui_compare_results.py 命令"""
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        webui_script_path = os.path.join(current_dir, "webui_compare_results.py")

        if not os.path.exists(webui_script_path):
            QMessageBox.critical(self, "错误", f"Web UI 脚本未找到: {webui_script_path}")
            self.terminal_output.appendHtml(f'<span style="color:red;">错误: Web UI 脚本未找到: {webui_script_path}</span>')
            return

        command = ["streamlit", "run", webui_script_path]
        
        self.terminal_output.appendPlainText(f"尝试启动结果分析 Web UI: {' '.join(command)}")
        self.terminal_output.appendPlainText("-----------------------------------")

        try:
            # 使用 Popen 在后台启动 streamlit，不阻塞 GUI
            # 注意：Streamlit 的输出通常在它自己的终端/浏览器中，而不是这里
            subprocess.Popen(command, cwd=current_dir) 
            self.terminal_output.appendPlainText("Streamlit 进程已启动。请检查新的终端窗口或浏览器。")
        except FileNotFoundError:
             QMessageBox.critical(self, "错误", "未找到 'streamlit' 命令。请确保 Streamlit 已安装并配置在系统 PATH 中。")
             self.terminal_output.appendHtml(f'<span style="color:red;">错误: 未找到 \'streamlit\' 命令。请确保 Streamlit 已安装并配置在系统 PATH 中。</span>')
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动 Streamlit 失败: {e}")
            self.terminal_output.appendHtml(f'<span style="color:red;">启动 Streamlit 失败: {e}</span>')

    @pyqtSlot()
    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace').rstrip()
        
        # 判断是否是处理进度信息
        if "处理进度:" in text:
            self.terminal_output.appendHtml(f'<span style="color:lime;">{text}</span>')
        # 判断是否是失败信息
        elif "失败" in text or "错误" in text:
            self.terminal_output.appendHtml(f'<span style="color:red;">{text}</span>')
        else:
            # 其他正常输出保持白色
            self.terminal_output.appendPlainText(text)
        
        # 自动滚动到底部
        self.terminal_output.verticalScrollBar().setValue(self.terminal_output.verticalScrollBar().maximum())
    
    @pyqtSlot()
    def handle_stderr(self):
        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf-8', errors='replace')
        # 使用红色显示错误信息
        self.terminal_output.appendHtml(f'<span style="color:red;">{text.rstrip()}</span>')
        # 自动滚动到底部
        self.terminal_output.verticalScrollBar().setValue(self.terminal_output.verticalScrollBar().maximum())
    
    @pyqtSlot(int, QProcess.ExitStatus)
    def process_finished(self, exit_code, exit_status):
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.terminal_output.appendPlainText("\n程序执行完成！")
            QMessageBox.information(self, "完成", "对比程序运行完成！")
        else:
            self.terminal_output.appendHtml(f'<span style="color:red;">\n程序执行失败，退出代码: {exit_code}</span>')
            QMessageBox.critical(self, "错误", f"执行失败，退出代码: {exit_code}")

if __name__ == "__main__":
    app = QApplication([])
    window = CompareUI()
    window.show()
    app.exec()