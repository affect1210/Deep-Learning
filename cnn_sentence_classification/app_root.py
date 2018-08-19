#!/usr/bin/env python
# -*- coding:utf-8 -*

import os

def get_root_path():
    # 返回path规范化的绝对路径
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    print(get_root_path())
