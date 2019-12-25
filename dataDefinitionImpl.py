# 作者 : lxw
# 文件 : dataDefinitionImpl.py
# 日期 : 2019/12/6 下午5:49
# IDE : PyCharm
# Description : DataDefinition 实现类
# Github : https://github.com/liuxw123

from DataOprt.dataset import DataSet
from values.values import NUM_TIMES, NUM_ANTENNA, IS_COMPLEX, NUM_POINT
from dataDefinition import DataDefinition
from modelConfig import DELIMITER
from values.strings import VERSION_NOT_SUPPORTED, KEY_LOGGING_DATA_HANDLER, KEY_LOGGING_DATA_INPUT, \
    KEY_LOGGING_DATA_TARGET, KEY_LOGGING_DATA_TRAIN_RATE, KEY_LOGGING_DATA_SHUFFLE, KEY_LOGGING_DATA_CLASSES

import numpy as np


class DataDefinitionImplV0(DataDefinition):
    """
    TODO input right key

    v0： version. 说明这是一个多分类定义模型
    3 ： subVersion. 三分类
    """

    def checkAllDotsUsedInformation(self):
        pass

    KEY = "v0-3-x-x"

    def details(self):
        """
        用于后续汇总输出信息时使用，
        :return: 返回出必要的记录信息
        """
        # TODO details
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__, KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (3 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}

        return info

    def __init__(self, train: float, shuffle=True) -> None:
        super().__init__()

        assert 0 < train < 1
        self.train = train  # 训练集比例
        self.dataset = DataSet().getData(dim=3)  # 264 * 10 * 16
        self.shuffle = shuffle
        self.classes = None

    @staticmethod
    def getUnused(labelInfo: list) -> list:
        """
        获取未使用到的数据
        :param labelInfo: 使用到的位置点分类数据
        :return:
        """
        unused = []

        for i in range(NUM_POINT):

            flag = True
            for clazz in labelInfo:
                if clazz.__contains__(i):
                    flag = False
                    break
            if flag:
                unused.append(i)

        return unused

    def checkKey(self, key1: str, key2: str) -> None:
        """
        check key,查看此类是否能用于目标Key
        :param key1: key1
        :param key2: key2
        :return:
        """
        # TODO check key
        version1, sub1, _, dataVersion1 = key1.split(DELIMITER)
        version2, sub2, _, dataVersion2 = key2.split(DELIMITER)

        if version1 == version2 and sub1 == sub2:
            return
        else:
            raise ValueError(VERSION_NOT_SUPPORTED)

    def getData(self, key: str) -> tuple:
        """
        见父类描述
        :param key: dst key. for check
        :return: 训练所需数据(xTrain, yTrain, xTest, yTest) if checked.
        """
        self.checkKey(key, DataDefinitionImplV0.KEY)
        return self.postProcess(self.getXY(self.labelDef()))

    def labelDef(self):
        """
        v0版本的定义位置点所属类
        :return: 分类结果
        """

        # TODO  labels decide by this

        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(157, 204)]
        class3 = [x for x in range(216, 264)]

        unused = self.getUnused([class1, class2, class3])

        self.classes = [class1, class2, class3, unused]

        return class1, class2, class3

    def getXY(self, labelInfo: tuple) -> tuple:
        """
        获取 input data & target data
        :param labelInfo: 分类信息数据
        :return:
        """
        labels = []
        inData = None
        for k, item in enumerate(labelInfo):
            data = np.ndarray((len(item) * NUM_TIMES, NUM_ANTENNA * IS_COMPLEX))
            for idx, num in enumerate(item):
                data[idx * NUM_TIMES:(idx + 1) * NUM_TIMES] = self.dataset[num]
                labels = labels + [k] * NUM_TIMES

            if inData is None:
                inData = data
            else:
                inData = np.vstack((inData, data))

        return inData, np.asarray(labels)

    def postProcess(self, data):
        """
        随机打乱数据，分出train & test 数据
        :param data: input data & target data
        :return:
        """
        xData, yData = data
        dim = xData.ndim
        nSample = xData.shape[0]
        train = int(self.train * nSample)

        random = np.random.permutation(nSample)
        if self.shuffle:
            shuffledXData = xData[random]
            shuffledYData = yData[random]
        else:
            shuffledXData = xData
            shuffledYData = yData

        if dim == 2:
            trainXData = shuffledXData[:train]
            trainYData = shuffledYData[:train]
            testXData = shuffledXData[train:]
            testYData = shuffledYData[train:]

        return trainXData, trainYData, testXData, testYData


class DataDefinitionImplV0D1(DataDefinitionImplV0):
    """
        TODO input right key

        v0： version. 说明这是一个多分类定义模型
        4 ： subVersion. 4分类
        """
    KEY = "v0-4-x-x"

    def details(self):
        """
        用于后续汇总输出信息时使用，
        :return: 返回出必要的记录信息
        """
        # TODO details
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__, KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (4 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}

        return info

    def getData(self, key: str) -> tuple:
        """
        见父类描述
        :param key: dst key. for check
        :return: 训练所需数据(xTrain, yTrain, xTest, yTest) if checked.
        """
        self.checkKey(key, DataDefinitionImplV0D1.KEY)
        return self.postProcess(self.getXY(self.labelDef()))

    def labelDef(self):
        """
        v0版本的定义位置点所属类
        :return: 分类结果
        """

        # TODO  labels decide by this

        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(63, 93)]
        class3 = [x for x in range(157, 204)]
        class4 = [x for x in range(216, 264)]

        unused = []

        for i in range(NUM_POINT):

            if class1.__contains__(i):
                continue

            if class2.__contains__(i):
                continue

            if class3.__contains__(i):
                continue

            if class4.__contains__(i):
                continue

            unused.append(i)

        self.classes = [class1, class2, class3, class4, unused]

        return class1, class2, class3, class4


class DataDefinitionImplV0D2(DataDefinitionImplV0):
    """
        TODO input right key

        v0： version. 说明这是一个多分类定义模型
        4 ： subVersion. 4分类
        """
    KEY = "v0-5-x-x"

    def details(self):
        """
        用于后续汇总输出信息时使用，
        :return: 返回出必要的记录信息
        """
        # TODO details
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__, KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (5 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}

        return info

    def getData(self, key: str) -> tuple:
        """
        见父类描述
        :param key: dst key. for check
        :return: 训练所需数据(xTrain, yTrain, xTest, yTest) if checked.
        """
        self.checkKey(key, DataDefinitionImplV0D2.KEY)
        return self.postProcess(self.getXY(self.labelDef()))

    def labelDef(self):
        """
        v0版本的定义位置点所属类
        :return: 分类结果
        """

        # TODO  labels decide by this

        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(63, 93)]
        class3 = [x for x in range(93, 101)]
        for x in range(111, 127):
            class3.append(x)
        class4 = [x for x in range(145, 204)]
        class5 = [x for x in range(216, 264)]

        unused = []

        for i in range(NUM_POINT):

            if class1.__contains__(i):
                continue

            if class2.__contains__(i):
                continue

            if class3.__contains__(i):
                continue

            if class4.__contains__(i):
                continue

            if class5.__contains__(i):
                continue

            unused.append(i)

        self.classes = [class1, class2, class3, class4, class5, unused]

        return class1, class2, class3, class4, class5
