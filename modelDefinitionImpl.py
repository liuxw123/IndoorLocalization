# 作者 : lxw
# 文件 : modelDefinitionImpl.py
# 日期 : 2019/12/9 上午9:53
# IDE : PyCharm
# Description : modelDefinition的实现类
# Github : https://github.com/liuxw123

from modelDefinition import ModelInterface
from modelConfig import DELIMITER, LAYER, DROPOUT_LAYER, DROPOUT_PARAMETER
from values.strings import VERSION_NOT_SUPPORTED, KEY_LOGGING_MODEL_HANDLER, KEY_LOGGING_MODEL_HIDDEN_LAYER, \
    KEY_LOGGING_MODEL_LAYERS

from torch import nn


class PstModelV0(ModelInterface):
    """
    TODO input right key

    v0: version. 说明这是一个多分类定义模型
    0 : modelVersion. 模型为第0次定义
    """
    KEY = "v0-x-0-x"

    def hiddenLayerString(self) -> list:

        cons = []
        for layer in self.model:
            cons.append(type(layer).__name__)

        return cons

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
        info = {KEY_LOGGING_MODEL_HANDLER: type(self).__name__, KEY_LOGGING_MODEL_HIDDEN_LAYER: LAYER,
                KEY_LOGGING_MODEL_LAYERS: self.hiddenLayerString()}
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


class PstModelV0M1(PstModelV0):
    """
        TODO input right key

        v0: version. 说明这是一个多分类定义模型
        1 : modelVersion. 模型为第1次定义
        """
    KEY = "v0-x-1-x"

    def checkKey(self, key: str) -> None:

        version1, _, modelVersion1, _ = key.split(DELIMITER)
        version2, _, modelVersion2, _ = PstModelV0M1.KEY.split(DELIMITER)

        # TODO check key
        if version1 == version2 and modelVersion1 == modelVersion2:
            return
        else:
            raise ValueError(VERSION_NOT_SUPPORTED)

    def creatModel(self, key: str) -> None:
        self.checkKey(key)

        hidden = LAYER
        drop = DROPOUT_LAYER
        dropPara = DROPOUT_PARAMETER

        part = []

        for i in range(len(hidden) - 2):
            inChn = hidden[i]
            outChn = hidden[i + 1]

            if i in drop:
                part.append(nn.Dropout(dropPara[drop.index(i)]))
            part.append(nn.Linear(inChn, outChn))
            part.append(nn.ReLU())

        part.append(nn.Linear(hidden[-2], hidden[-1]))
        part.append(nn.Softmax(dim=1))

        self.model = nn.ModuleList(part)
