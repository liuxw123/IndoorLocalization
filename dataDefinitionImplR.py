# 作者 : lxw
# 文件 : dataDefinitionImplR.py
# 日期 : 2019/12/13 下午3:18
# IDE : PyCharm
# Description : DataDefinition 实现类 与dataDefinitionImpl.py不同的是，在于选取测试集，将某个位置点的整次采集都作为测试
# Github : https://github.com/liuxw123

from dataDefinitionImpl import DataDefinitionImplV0
import numpy as np

from values.strings import KEY_LOGGING_DATA_HANDLER, KEY_LOGGING_DATA_TARGET, KEY_LOGGING_DATA_SHUFFLE, \
    KEY_LOGGING_DATA_INPUT, KEY_LOGGING_DATA_TRAIN_RATE, KEY_LOGGING_DATA_CLASSES


class DataDefinitionImplRV0(DataDefinitionImplV0):

    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__, KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (3 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def divide(self, labelInfo: list) -> tuple:

        train = []
        test = []

        for clazz in labelInfo:
            num = len(clazz)
            numTest = int(num * (1 - self.train))
            trainClass = clazz
            testClass = []
            while numTest > 0:
                n1 = np.random.randint(num)
                testClass.append(trainClass.pop(n1))
                num -= 1
                numTest -= 1
            train.append(trainClass)
            test.append(testClass)

        train.sort()
        test.sort()

        return train, test

    def labelDef(self):
        return super().labelDef()

    def getTrainAndTestClassesDots(self):
        classes = self.labelDef()

        unused = self.getUnused(classes)

        train, test = self.divide(classes)

        self.classes = []
        for i in range(len(train)):
            self.classes.append([train[i], test[i]])
        self.classes.append(unused)

        return train, test

    def getData(self, key: str) -> tuple:

        self.checkKey(key, DataDefinitionImplRV0.KEY)

        train, test = self.getTrainAndTestClassesDots()

        trainXData, trainYData = self.postProcess(self.getXY(train))
        testXData, testYData = self.postProcess(self.getXY(test))
        return trainXData, trainYData, testXData, testYData

    def postProcess(self, data):
        xData, yData = data
        nSample = xData.shape[0]

        random = np.random.permutation(nSample)
        if self.shuffle:
            shuffledXData = xData[random]
            shuffledYData = yData[random]
        else:
            shuffledXData = xData
            shuffledYData = yData

        return shuffledXData, shuffledYData


class DataDefinitionImplRV0S0(DataDefinitionImplRV0):
    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__,
                KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (3 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def labelDef(self):
        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(133, 204)]
        class3 = [x for x in range(216, 264)]
        return [class1, class2, class3]


class DataDefinitionImplRV0S1(DataDefinitionImplRV0):
    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__,
                KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (3 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def labelDef(self):
        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(127, 204)]
        class3 = [x for x in range(216, 264)]
        return [class1, class2, class3]


class DataDefinitionImplRV0D1(DataDefinitionImplRV0):
    KEY = "v0-4-x-x"

    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__,
                KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (4 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def getData(self, key: str) -> tuple:
        self.checkKey(key, DataDefinitionImplRV0D1.KEY)

        train, test = self.getTrainAndTestClassesDots()

        trainXData, trainYData = self.postProcess(self.getXY(train))
        testXData, testYData = self.postProcess(self.getXY(test))
        return trainXData, trainYData, testXData, testYData

    def labelDef(self):
        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(63, 106)]
        class3 = [x for x in range(127, 204)]
        class4 = [x for x in range(216, 264)]
        return [class1, class2, class3, class4]


class DataDefinitionImplRV0D1S0(DataDefinitionImplRV0D1):
    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__,
                KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (4 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def labelDef(self):
        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(61, 109)]
        class3 = [x for x in range(127, 204)]
        class4 = [x for x in range(216, 264)]
        return [class1, class2, class3, class4]


class DataDefinitionImplRV0D1S1(DataDefinitionImplRV0D1):

    def details(self):
        info = {KEY_LOGGING_DATA_HANDLER: type(self).__name__,
                KEY_LOGGING_DATA_INPUT: "CSI data(dimension: 16[8*2]).",
                KEY_LOGGING_DATA_TARGET: "one-hot (4 classes).", KEY_LOGGING_DATA_TRAIN_RATE: self.train,
                KEY_LOGGING_DATA_SHUFFLE: self.shuffle, KEY_LOGGING_DATA_CLASSES: self.classes}
        return info

    def labelDef(self):
        class1 = [x for x in range(0, 61)]
        class2 = [x for x in range(63, 109)]
        class3 = [x for x in range(133, 213)]
        class4 = [x for x in range(216, 264)]
        return [class1, class2, class3, class4]
