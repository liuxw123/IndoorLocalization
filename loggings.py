# 作者 : lxw
# 文件 : loggings.py
# 日期 : 2019/12/9 上午11:38
# IDE : PyCharm
# Description : 记录训练相关信息
# Github : https://github.com/liuxw123
from abc import ABCMeta, abstractmethod

from commonInterface import Common


class ResultLog(Common, metaclass=ABCMeta):

    def __init__(self, key: str, debug=False) -> None:
        super().__init__()
        self.checkKey(key)
        self.log = {}
        self.logStr = ""
        self.debug = debug


    def details(self):
        pass

    def write(self, filePath):

        if self.debug:
            return

        fw = open(filePath, "w+")
        fw.writelines(self.logStr)
        fw.close()

    @abstractmethod
    def logging(self) -> None:
        """
        记录本次运行测试信息
        :return:
        """
        pass

    def add(self, key: str, val: dict) -> None:
        """
        添加记录字段， key-value
        :param key:
        :param val:
        :return:
        """
        assert key in ["Data", "Model", "Result"]

        if key in self.log.keys():
            val1 = self.log[key]
            self.log[key] = {**val1, **val}
        else:
            self.log[key] = val
