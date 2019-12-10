# 作者 : lxw
# 文件 : utils.py
# 日期 : 2019/12/2 15:17
# IDE : PyCharm
# Description : 一些小工具函数
# Github : https://github.com/liuxw123

import os
import re

from values.strings import FILE_DELIMITER


def readFile(file) -> list:
    """
    读取文件

    :param file: 文件路径
    :return: List<String>
    """
    try:
        fr = open(file)
        contents = fr.readlines()
        fr.close()
        return contents
    except Exception:
        print("打开文件{}失败".format(file))
        return []


def stripBlankSpace(string: str, mode=0) -> str:
    """
    去除字符串首尾空格或换行符

    :param string: 目标字符串
    :param mode:
        0 : 首尾都去除
        1 : 去除头部
        2 : 去除尾部
    :return: 处理后的字符串
    """
    assert mode in [0, 1, 2]

    if mode == 0:
        return string.strip()
    elif mode == 1:
        cnt = 0
        while cnt < len(string) and (string[cnt] == " " or string[cnt] == "\n"):
            cnt += 1

        return string[cnt:]
    else:
        cnt = len(string) - 1
        while cnt > 0 and (string[cnt] == " " or string[cnt] == "\n"):
            cnt -= 1

        return string[:cnt + 1]


def arrayString(arr: list, connectChar=", ") -> str:
    """
    实数序列字符串
    :param arr: 实数序列
    :param connectChar: 数字间的连接符
    :return: 字符串
    """
    string = "["

    for num in arr:
        string += "{}".format(num) + connectChar
    try:
        string = string[:-2] + "]"
    except Exception:
        string = "[]"
    return string


def match(pattern, string):
    ans = re.search(pattern, string)

    return ans is not None


def getDirectory(root):
    files = []
    for file in os.listdir(root):
        if os.path.isdir(root + FILE_DELIMITER + file) and match(r'^[0-9a-zA-Z]+$', file):
            files.append(file)
    return files




