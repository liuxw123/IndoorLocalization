# 作者 : lxw
# 文件 : modelDefinitionImpl.py
# 日期 : 2019/12/9 上午9:53
# IDE : PyCharm
# Description : modelDefinition的实现类
# Github : https://github.com/liuxw123

from modelDefinition import ModelInterface
from modelConfig import DELIMITER, LAYER
from values.strings import VERSION_NOT_SUPPORTED

from torch import nn


class PstModelV0(ModelInterface):
    """
    TODO input right key

    v0: version. 说明这是一个多分类定义模型
    0 : modelVersion. 模型为第0次定义
    """
    KEY = "v0-x-0-x"

    def __init__(self, key: str) -> None:
        """
        initial
        :param key: dst key, for check key.
        """
        super().__init__()
        self.creatModel(key)

    def creatModel(self, key: str) -> None:
        """
        创建model
        :param key: dst key, for check key.
        :return:
        """
        self.checkKey(key)

        hidden = LAYER
        part = []

        for i in range(len(hidden) - 2):
            inChn = hidden[i]
            outChn = hidden[i + 1]
            part.append(nn.Linear(inChn, outChn))
            part.append(nn.ReLU())

        part.append(nn.Linear(hidden[-2], hidden[-1]))
        part.append(nn.Softmax(dim=1))

        self.model = nn.ModuleList(part)

    def checkKey(self, key: str) -> None:

        version1, _, modelVersion1, _ = key.split(DELIMITER)
        version2, _, modelVersion2, _ = PstModelV0.KEY.split(DELIMITER)

        # TODO check key
        if version1 == version2 and modelVersion1 == modelVersion2:
            return
        else:
            raise ValueError(VERSION_NOT_SUPPORTED)

    def details(self):
        # TODO details
        info = {"name": type(self).__name__, "hidden layer": LAYER}
        return info

    def forward(self, x):
        """
        forward propagation
        :param x:
        :return:
        """
        for layer in self.model:
            x = layer(x)

        return x
