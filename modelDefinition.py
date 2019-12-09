# 作者 : lxw
# 文件 : modelDefinition.py
# 日期 : 2019/12/4 18:35
# IDE : PyCharm
# Description : 定义 NN
# Github : https://github.com/liuxw123
from abc import ABCMeta, abstractmethod

from torch import nn
from commonInterface import Common


class ModelInterface(nn.Module, Common, metaclass=ABCMeta):
    """
    model的公共接口类
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def creatModel(self, key: str) -> None:
        """
        创建model
        :param key: dst key, for check key.
        :return:
        """
        pass
