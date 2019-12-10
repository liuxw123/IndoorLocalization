# 作者 : lxw
# 文件 : plot.py
# 日期 : 2019/12/2 20:32
# IDE : PyCharm
# Description : 位置点坐标可视化
# Github : https://github.com/liuxw123
import numpy as np
import matplotlib.pyplot as plt

from DataOprt.utils import readFile, stripBlankSpace
from values.strings import PATH_POSITION_DATA
from values.values import NUM_POINT


class PstDataSet:
    def __init__(self) -> None:
        super().__init__()

        self.pstData = np.ndarray((NUM_POINT, 2))
        self.read()

    def read(self) -> None:
        """
        读取采集的位置点坐标文件
        :return:
        """
        data = readFile(PATH_POSITION_DATA)

        for idx, item in enumerate(data):
            string = stripBlankSpace(item).split(" ")
            xPos = int(string[0])
            yPos = int(string[1])
            self.pstData[idx, 0] = xPos
            self.pstData[idx, 1] = yPos

    def plotCommon(self):

        string = "-" * 42 + "\n"
        string += "|" + " " * 5 + "0 -- 60: 研发区域" + " " * 18 + "|\n"
        string += "|" + " " * 5 + "61 62: 楼梯间过道" + " " * 18 + "|\n"
        string += "|" + " " * 5 + "63 -- 84: 前台处" + " " * 19 + "|\n"
        string += "|" + " " * 5 + "85 -- 92: 茶水间" + " " * 19 + "|\n"
        string += "|" + " " * 5 + "93 -- 105: 小会议室" + " " * 16 + "|\n"
        string += "|" + " " * 5 + "106 -- 110: 会议室旁过道" + " " * 11 + "|\n"
        string += "|" + " " * 5 + "111 -- 126: 大会议室" + " " * 15 + "|\n"
        string += "|" + " " * 5 + "127 -- 156: 研发区域" + " " * 15 + "|\n"
        string += "|" + " " * 5 + "157 -- 203: 视听区域与小洽谈室" + " " * 5 + "|\n"
        string += "|" + " " * 5 + "204 -- 215: 财务/总经理办公室" + " " * 6 + "|\n"
        string += "|" + " " * 5 + "216 -- 263: 行政办公区域" + " " * 11 + "|\n"
        string += "-" * 42 + "\n"
        plt.text(4200, 1200, string, fontsize=12)

        plt.plot([self.pstData[60, 0] + 75, self.pstData[60, 0] + 75], [-50, self.pstData[60, 1] + 100])
        plt.plot([self.pstData[152, 0] + 135, self.pstData[152, 0] + 135], [-50, self.pstData[146, 1] + 100])
        plt.plot([self.pstData[181, 0] + 135, self.pstData[181, 0] + 135],
                 [self.pstData[110, 1] - 75, self.pstData[109, 1] + 50])
        plt.plot([self.pstData[110, 0] + 100, self.pstData[110, 0] + 100],
                 [self.pstData[110, 1] - 75, self.pstData[106, 1] + 100])
        plt.plot([self.pstData[180, 0] + 160, self.pstData[111, 0] + 135],
                 [self.pstData[110, 1] - 75, self.pstData[110, 1] - 75])
        plt.plot([self.pstData[123, 0] - 75, self.pstData[114, 0] + 160],
                 [self.pstData[123, 1] + 100, self.pstData[123, 1] + 100])
        plt.plot([self.pstData[201, 0] + 135, self.pstData[201, 0] + 135],
                 [self.pstData[201, 1] - 135, self.pstData[94, 1] + 100])
        plt.plot([self.pstData[215, 0] + 135, self.pstData[215, 0] + 135],
                 [self.pstData[220, 1] - 135, self.pstData[213, 1] + 100])

        plt.plot(self.pstData[127, 0] + 100, self.pstData[127, 1] - 100, "r*", markersize=20)
        plt.annotate("AP", xy=(self.pstData[127, 0] + 150, self.pstData[127, 1] - 150),
                     xytext=(self.pstData[127, 0] + 600, self.pstData[127, 1] - 500),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        plt.axis("equal")
        plt.xlim((-50, 50 + np.max(self.pstData[:, 0])))
        plt.ylim((-50, 50 + np.max(self.pstData[:, 1])))

    def plot(self) -> None:
        """
        打印采集点坐标图
        :return:
        """

        plt.figure(figsize=(13, 6.5))

        for i in range(NUM_POINT):
            x = self.pstData[i, 0]
            y = self.pstData[i, 1]
            plt.plot(x, y, "bx")

            if i >= 10:
                plt.text(x - 40, y + 15, i, fontsize=12)
            elif i >= 100:
                plt.text(x - 0, y + 15, i, fontsize=12)
            else:
                plt.text(x - 15, y + 15, i, fontsize=12)

        self.plotCommon()
        plt.savefig("plot.png")
        plt.show()

    def plotMultiClasses(self, classes: list, file) -> None:
        fig = plt.figure(figsize=(13, 6.5))
        colors = ["r", "b", "y", "g", "k"]
        dotShape = "x"

        for i, clazz in enumerate(classes):
            if i == len(classes) - 1:
                color = colors[-1]
                label = "unused"
            else:
                color = colors[i]
                label = "class {}".format(i)
            marker = color + dotShape

            num = clazz[0]
            x = self.pstData[num, 0]
            y = self.pstData[num, 1]
            plt.plot(x, y, marker, label=label)

            if num >= 10:
                plt.text(x - 40, y + 15, num, fontsize=12, color=color)
            elif num >= 100:
                plt.text(x - 0, y + 15, num, fontsize=12, color=color)
            else:
                plt.text(x - 15, y + 15, num, fontsize=12, color=color)

            for j in range(1, len(clazz)):
                num = clazz[j]
                x = self.pstData[num, 0]
                y = self.pstData[num, 1]
                plt.plot(x, y, marker)

                if num >= 10:
                    plt.text(x - 40, y + 15, num, fontsize=12, color=color)
                elif num >= 100:
                    plt.text(x - 0, y + 15, num, fontsize=12, color=color)
                else:
                    plt.text(x - 15, y + 15, num, fontsize=12, color=color)

            plt.legend(loc="upper left")

        self.plotCommon()
        plt.savefig(file, bbox_inches='tight', pad_inches=0)
        # plt.show()

# usage
# dataset = PstDataSet()
# dataset.plot()
