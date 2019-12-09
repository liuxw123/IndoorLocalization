# 作者 : lxw
# 文件 : dataset.py
# 日期 : 2019/12/2 11:49
# IDE : PyCharm
# Description : 处理原始采集数据的类
# Github : https://github.com/liuxw123


from DataOprt.utils import readFile, stripBlankSpace

import numpy as np

from values.strings import PATH_ORIG_DATA
from values.values import NUM_TIMES, NUM_POINT, NUM_ANTENNA, IS_COMPLEX, NUM_DETECTION_USER


class DataSet:

    def __init__(self) -> None:
        super().__init__()

        self.holder = DataSet.Compute()

        # usage
        # data = self.holder.index(self.getNum(0, 0))
        # corr = self.holder.onePointCorrelation(0)
        print("Dataset read OK!")

    @staticmethod
    def getNum(idx: int, whichTime: int) -> int:
        """
        将位置点序号和采集序号转换为序号索引
        :param idx: 位置点序号 should be 0 <= idx < 264
        :param whichTime: 10次采集中的哪一次
        :return:
        """
        return idx * NUM_TIMES + whichTime

    @staticmethod
    def whichFile(idx: int, whichTime: int) -> str:
        """

        :param idx: 位置点序号 should be 0 <= idx < 264
        :param whichTime: 10次采集中的哪一次
        :return: 文件名称
        """
        return PATH_ORIG_DATA + "/{}.txt".format(idx * NUM_TIMES + whichTime + 1)

    def getData(self, dim=3) -> np.ndarray:
        """
        获取所有的数据
        :return: data shape: 2640*16
        """
        assert dim in [2, 3]

        if dim == 3:
            return self.holder.convertDimensionData()
        else:
            return self.holder.getAllData()

    class Compute:

        def __init__(self) -> None:
            super().__init__()

            self.reader = DataSet.Reader()
            self.reader.read()
            # 采集数据中UserDetection输出的用户数据，选取最大能量的用户数据 shape: 264*10 16
            self.pointData = np.ndarray((NUM_POINT * NUM_TIMES, NUM_ANTENNA * IS_COMPLEX))
            self.getPointData()

        @staticmethod
        def computeEnergy(data: np.ndarray) -> float:
            """
            计算能量
            :param data: 数据
            :return: 能量值
            """
            return np.sqrt(data.dot(data.conj()).real)

        def getOnePointDataEnergy(self, idx) -> np.ndarray:
            """
            获取一个位置点20个用户的能量值
            :param idx: 位置点序号
            :return: List<float> 各用户能量值
            """
            ans = []
            for i in range(NUM_DETECTION_USER):
                eny = self.computeEnergy(self.convertToComplex(self.reader.index(idx, i)))
                ans.append(eny)

            return np.asarray(ans)

        def getPointData(self) -> None:
            """
            获取位置点，最大能量的那个用户数据,存储到 self.pointData
            :return:
            """
            for i in range(NUM_POINT):
                for j in range(NUM_TIMES):
                    num = DataSet.getNum(i, j)
                    eny = self.getOnePointDataEnergy(num)
                    maxUserId = np.argmax(eny)
                    self.pointData[num] = self.reader.index(num, maxUserId)

        @staticmethod
        def convertToComplex(data: np.ndarray) -> np.ndarray:
            """
            将实数序列变为复数序列
            :param data: 数据
            :return: 转换后的数据
            """
            real = data[0::2]
            imag = data[1::2]

            return real + (1j * imag)

        def index(self, idx: int) -> np.ndarray:
            """
            获取某个位置点的数据,某次采集的数据（具有最大能量的那个用户数据）
            :param idx: 位置点序号, should be 0 <= idx < 264*10
            :return: 返回该位置点的数据
            """
            assert 0 <= idx < NUM_POINT * NUM_TIMES
            return self.pointData[idx]

        def indexOnePointData(self, idx: int) -> np.ndarray:
            """
            获取某个位置点的数据,10次采集的数据（具有最大能量的那个用户数据）
            :param idx: 位置点序号, should be 0 <= idx < 264
            :return: 返回该位置点的数据
            """
            assert 0 <= idx < NUM_POINT
            return self.pointData[idx * NUM_TIMES:(idx + 1) * NUM_TIMES]

        def convertDimensionData(self) -> np.ndarray:
            """
            将 2640*16的数据转换为三维数据, 264*10*16
            :return: 转换后的数据
            """
            ans = np.ndarray((NUM_POINT, NUM_TIMES, NUM_ANTENNA * IS_COMPLEX))

            for i in range(NUM_POINT):
                ans[i] = self.pointData[i * NUM_TIMES:(i + 1) * NUM_TIMES]
            return ans

        def getDataEnergy(self, idx) -> float:
            """
            获取能量
            :param idx: 位置点序号, should be 0 <= idx < 264*10
            :return:
            """
            assert 0 <= idx < NUM_POINT * NUM_TIMES
            return self.computeEnergy(self.pointData[idx])

        @staticmethod
        def correlation(data1: np.ndarray, data2: np.ndarray) -> float:
            """
            相关系数计算
            :param data1: 第一个序列
            :param data2: 第二个序列
            :return: 相关系数值
            """
            return np.abs(data1.dot(data2.conj())) / np.sqrt((np.abs(data1.dot(data1.conj())))) / np.sqrt(
                (np.abs(data2.dot(data2.conj()))))

        def onePointCorrelation(self, idx) -> np.ndarray:
            """
            获取一个位置点不同采集序号的相关系数矩阵
            :param idx: 位置点编号 should be 0 <= idx < 264
            :return: 相关系数矩阵
            """
            assert 0 <= idx < NUM_POINT
            ans = np.ndarray((NUM_TIMES, NUM_TIMES))

            for i in range(NUM_TIMES):
                for j in range(NUM_TIMES):
                    if i == j:
                        ans[i, j] = 1
                        continue
                    ans[i, j] = self.correlation(self.convertToComplex(self.pointData[DataSet.getNum(idx, i)]),
                                                 self.convertToComplex(self.pointData[DataSet.getNum(idx, j)]))

            return ans

        def getCorrelation(self, idx1: int, idx2: int) -> float:
            """
            获取两个不同点的相关系数
            :param idx1: 第一个点
            :param idx2: 第二个点
            :return: 相关系数值
            """

            return self.correlation(self.pointData[idx1], self.pointData[idx2])

        def getAllData(self) -> np.ndarray:
            """
            获取所有的数据
            :return: data shape: 2640*16
            """

            return self.pointData

    class Reader:
        """
        读取原始采集数据的类
        """

        def __init__(self) -> None:
            super().__init__()
            # 原始文件中读出的数据
            self.origData = np.ndarray(
                (NUM_POINT * NUM_TIMES, NUM_DETECTION_USER, NUM_ANTENNA * IS_COMPLEX))  # shape: 264*10 20 16

        def read(self) -> None:
            """
            读取原始文件中的数据到 self.origData shape: 264 20 16
            :return: None
            """

            for i in range(NUM_POINT):
                for j in range(NUM_TIMES):

                    num = DataSet.getNum(i, j)
                    file = PATH_ORIG_DATA + "/{}.txt".format(num + 1)
                    dataStr = readFile(file)
                    data = []

                    for item in dataStr:
                        data.append(float(stripBlankSpace(item)))

                    data = np.asarray(data)

                    for k in range(NUM_DETECTION_USER):
                        self.origData[num, k] = data[
                                                k * (NUM_ANTENNA * IS_COMPLEX):(k + 1) * (NUM_ANTENNA * IS_COMPLEX)]

        def getOne(self, idx: int) -> np.ndarray:
            """
            获取一个位置点采集到的数据
            :param idx: 位置点序号 should be 0 <= idx < 264*10
            :return: 返回该位置点采集的数据
            """
            assert 0 <= idx < NUM_POINT * NUM_TIMES

            return self.origData[idx]

        def index(self, idx: int, userIdx: int) -> np.ndarray:
            """
            获取一个位置点的UserDetection输出的用户响应数据
            :param idx: 位置点序号 should be 0 <= idx < 264*10
            :param userIdx: 用户序号 should be 0 <= userIdx < 20
            :return: 返回位置点的UserDetection输出的用户响应数据
            """
            assert 0 <= idx < NUM_POINT * NUM_TIMES
            assert 0 <= userIdx < NUM_DETECTION_USER

            return self.origData[idx, userIdx]
