# 作者 : lxw
# 文件 : dataDefinition.py
# 日期 : 2019/12/3 17:30
# IDE : PyCharm
# Description : 接口， 定义数据如何制作的接口
# Github : https://github.com/liuxw123

from abc import abstractmethod, ABCMeta

from commonInterface import Common


class DataDefinition(Common, metaclass=ABCMeta):
    """
    定义数据如何制作，都应该继承此类
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def getData(self) -> tuple:
        """
        获取训练所需数据， 调用此函数即可获取，具体返回出数据形式见子类说明
        :return: 返回训练集数据，测试集数据
        """
        pass

    @abstractmethod
    def postProcess(self):
        """
        数据准备即将完成的最后一步，一般地，执行完此函数后，即得到所需的训练数据，具体操作见子类实现
        :return: 返回经过此函数处理后的数据，即训练数据
        """
        pass

    @abstractmethod
    def checkAllDotsUsedInformation(self):
        """
        核对数据集
        :return:
        """
        pass
