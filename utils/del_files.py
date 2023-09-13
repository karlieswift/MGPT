"""
@Env: /anaconda3/python3.10
@Time: 2023/7/14-18:25
@Auth: karlieswift
@File: del_files.py
@Desc: 
"""

import os
def del_files(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))