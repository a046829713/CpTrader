import subprocess
import time
import os
from utils.Debug_tool import debug
import logging


# =============================================================================================
# 获取当前文件夹的绝对路径
current_folder = os.getcwd()


# 设置要运行的可执行文件路径
exe_path = os.path.join(current_folder, 'TradingSysteam.exe')



process = subprocess.Popen([exe_path])

while True:
    if process.poll() is None:
        pass
    else:
        debug.record_msg("System crashed, restart the program.", log_level=logging.error)
        process = subprocess.Popen([exe_path, '--autostart'])

    time.sleep(30)

