# 作者 : lxw
# 文件 : trainData.py
# 日期 : 2019/12/4 17:10
# IDE : PyCharm
# Description : 训练时所需要的数据集，继承于torch.utils.data.Dataset
# Github : https://github.com/liuxw123

from torch.utils.data import Dataset
from dataDefinitionImplR import DataDefinitionImplRV0D1S1

import torch
import numpy as np


class TrainData(Dataset):
    """
    use for train dataset.
    """

    def __init__(self, trainRate: float, key: str) -> None:
        super().__init__()

        # TODO choose right data process class
        self.dataHolder = DataDefinitionImplRV0D1S1(trainRate)

        self.xTrain, self.yTrain, self.xTest, self.yTest = self.dataHolder.getData(key)

        print("train data:{} samples".format(self.xTrain.shape[0]))
        print("test data:{} samples".format(self.xTest.shape[0]))

        # training phase: train or test
        self.phase = None
        # data for training
        self.xData = None
        self.yData = None
        # number of samples
        self.samples = 0

        # default is train phase
        self.changeToTrainPhase()

    def change(self, phase: str) -> None:
        """
        变更phase
        :param phase: string. train or test
        :return:
        """
        assert phase in ["train", "test"]

        self.phase = phase

        if self.phase == "train":
            self.xData = self.xTrain
            self.yData = self.yTrain
        else:
            self.xData = self.xTest
            self.yData = self.yTest

        self.samples = self.yData.shape[0]

    def changeToTrainPhase(self) -> None:
        """
        变更为 train phase
        :return:
        """
        self.change("train")

    def changeToTestPhase(self) -> None:
        """
        变更为 test phase
        :return:
        """
        self.change("test")

    def __getitem__(self, index: int):

        assert 0 <= index < self.samples
        return self.xData[index], self.yData[index]

    def __len__(self) -> int:
        return self.samples


def collate(batch):
    """
    batch data 提交
    :param batch: batch data
    :return:
    """
    xd = None
    yd = None
    for item in batch:
        x, y = item

        if xd is None:
            xd = x
            yd = y
        else:
            xd = np.vstack((xd, x))
            yd = np.hstack((yd, y))

    return torch.tensor(xd, dtype=torch.float), torch.tensor(yd, dtype=torch.long)
