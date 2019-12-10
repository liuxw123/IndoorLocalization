# 作者 : lxw
# 文件 : loggingImpl.py
# 日期 : 2019/12/9 下午2:47
# IDE : PyCharm
# Description : loggings的实现类
# Github : https://github.com/liuxw123
from DataOprt.plot import PstDataSet
from DataOprt.utils import arrayString, getDirectory
from loggings import ResultLog
from modelConfig import KEY
from values.strings import DELIMITER, VERSION_NOT_SUPPORTED, KEY_LOGGING_DATA, KEY_LOGGING_MODEL, KEY_LOGGING_RESULT, \
    EQUALS_DELIMITER, LINE_BREAK, KEY_LOGGING_DATA_HANDLER, CONTENT_DELIMITER, KEY_LOGGING_DATA_INPUT, \
    KEY_LOGGING_DATA_TARGET, KEY_LOGGING_DATA_TRAIN_RATE, KEY_LOGGING_DATA_SHUFFLE, KEY_LOGGING_DATA_CLASSES, \
    KEY_LOGGING_MODEL_HANDLER, KEY_LOGGING_MODEL_HIDDEN_LAYER, KEY_LOGGING_MODEL_BATCH, KEY_LOGGING_MODEL_OPTIMIZER, \
    KEY_LOGGING_MODEL_LR, KEY_LOGGING_MODEL_EPOCH, KEY_LOGGING_MODEL_LOSS_FUNCTION, KEY_LOGGING_RESULT_ACC, \
    KEY_LOGGING_RESULT_LOSS, PATH_RESULT, FILE_DELIMITER, PATH_RESULT_TXT, PATH_RESULT_MODEL, \
    KEY_LOGGING_RESULT_MODEL_PATH, KEY_LOGGING_RESULT_MODEL_PARAMETER, KEY_LOGGING_MODEL_LAYERS, UNDERLINE, \
    PATH_RESULT_PNG
from values.values import LOGGING_EQUALS_DELIMITER, NUM_POINT, LOGGING_BLANK_NUM

import os
import time
import torch


class LoggingImpl(ResultLog):
    """
    TODO input right key

    """
    KEY = "v0-3-0-0"

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.plot = PstDataSet()

    def loggingBefore(self):
        self.logStr += "key" + CONTENT_DELIMITER + KEY + LINE_BREAK
        self.logStr += "author" + CONTENT_DELIMITER + "lxw" + LINE_BREAK
        self.logStr += "time" + CONTENT_DELIMITER + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + LINE_BREAK
        self.logStr += LINE_BREAK

    def loggingData(self, plotFile: str):
        left = (LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_DATA)) // 2
        right = LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_DATA) - left
        self.logStr += EQUALS_DELIMITER * left + KEY_LOGGING_DATA + EQUALS_DELIMITER * right + LINE_BREAK

        info = self.log[KEY_LOGGING_DATA]
        self.logStr += KEY_LOGGING_DATA_HANDLER + CONTENT_DELIMITER + info[KEY_LOGGING_DATA_HANDLER] + LINE_BREAK
        self.logStr += KEY_LOGGING_DATA_INPUT + CONTENT_DELIMITER + info[KEY_LOGGING_DATA_INPUT] + LINE_BREAK
        self.logStr += KEY_LOGGING_DATA_TARGET + CONTENT_DELIMITER + info[KEY_LOGGING_DATA_TARGET] + LINE_BREAK
        self.logStr += KEY_LOGGING_DATA_TRAIN_RATE + CONTENT_DELIMITER + "{}".format(
            info[KEY_LOGGING_DATA_TRAIN_RATE]) + LINE_BREAK
        self.logStr += KEY_LOGGING_DATA_SHUFFLE + CONTENT_DELIMITER + "{}".format(
            info[KEY_LOGGING_DATA_SHUFFLE]) + LINE_BREAK
        self.logStr += KEY_LOGGING_DATA_CLASSES + CONTENT_DELIMITER + LINE_BREAK

        # TODO 画图显示
        classes = info[KEY_LOGGING_DATA_CLASSES]
        self.plot.plotMultiClasses(classes, plotFile)

        unused = classes.pop()
        for i, clazz in enumerate(classes):
            self.logStr += " " * LOGGING_BLANK_NUM + "class {}: ".format(i) + arrayString(clazz) + LINE_BREAK

        self.logStr += " " * LOGGING_BLANK_NUM + "unused: " + arrayString(unused) + LINE_BREAK

        self.logStr += EQUALS_DELIMITER * LOGGING_EQUALS_DELIMITER + LINE_BREAK
        self.logStr += LINE_BREAK

    def loggingModel(self):
        left = (LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_MODEL)) // 2
        right = LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_MODEL) - left
        self.logStr += EQUALS_DELIMITER * left + KEY_LOGGING_MODEL + EQUALS_DELIMITER * right + LINE_BREAK

        info = self.log[KEY_LOGGING_MODEL]

        self.logStr += KEY_LOGGING_MODEL_HANDLER + CONTENT_DELIMITER + info[KEY_LOGGING_MODEL_HANDLER] + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_HIDDEN_LAYER + CONTENT_DELIMITER + arrayString(
            info[KEY_LOGGING_MODEL_HIDDEN_LAYER], connectChar=DELIMITER + DELIMITER) + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_LAYERS + CONTENT_DELIMITER + arrayString(
            info[KEY_LOGGING_MODEL_LAYERS], connectChar=DELIMITER + DELIMITER) + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_BATCH + CONTENT_DELIMITER + "{}".format(
            info[KEY_LOGGING_MODEL_BATCH]) + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_OPTIMIZER + CONTENT_DELIMITER + info[KEY_LOGGING_MODEL_OPTIMIZER] + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_LR + CONTENT_DELIMITER + "{}".format(info[KEY_LOGGING_MODEL_LR]) + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_EPOCH + CONTENT_DELIMITER + "{}".format(
            info[KEY_LOGGING_MODEL_EPOCH]) + LINE_BREAK
        self.logStr += KEY_LOGGING_MODEL_LOSS_FUNCTION + CONTENT_DELIMITER + info[
            KEY_LOGGING_MODEL_LOSS_FUNCTION] + LINE_BREAK

        self.logStr += EQUALS_DELIMITER * LOGGING_EQUALS_DELIMITER + LINE_BREAK
        self.logStr += LINE_BREAK

    def loggingResult(self, modelPath):

        left = (LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_RESULT)) // 2
        right = LOGGING_EQUALS_DELIMITER - len(KEY_LOGGING_RESULT) - left
        self.logStr += EQUALS_DELIMITER * left + KEY_LOGGING_RESULT + EQUALS_DELIMITER * right + LINE_BREAK

        info = self.log[KEY_LOGGING_RESULT]
        self.logStr += KEY_LOGGING_RESULT_ACC + CONTENT_DELIMITER + "{:.2f}%".format(
            info[KEY_LOGGING_RESULT_ACC]) + LINE_BREAK
        self.logStr += KEY_LOGGING_RESULT_LOSS + CONTENT_DELIMITER + "{:.4f}".format(
            info[KEY_LOGGING_RESULT_LOSS]) + LINE_BREAK

        self.logStr += KEY_LOGGING_RESULT_MODEL_PATH + CONTENT_DELIMITER + modelPath + LINE_BREAK

        torch.save(info[KEY_LOGGING_RESULT_MODEL_PARAMETER], modelPath)

        self.logStr += EQUALS_DELIMITER * LOGGING_EQUALS_DELIMITER + LINE_BREAK
        self.logStr += LINE_BREAK

    @staticmethod
    def createResultDirectory():

        if not os.path.exists(PATH_RESULT):
            os.mkdir(PATH_RESULT)

        dirs = getDirectory(PATH_RESULT)
        total = {}

        for item in dirs:
            key = item.split(UNDERLINE)[0]

            try:
                total[key] += 1
            except Exception:
                total[key] = 1

        timeKey = time.strftime("%Y%m%d", time.localtime())

        try:
            num = total[timeKey]
        except Exception:
            num = 0

        return FILE_DELIMITER + timeKey + UNDERLINE + "{}".format(num)

    def logging(self):
        path = self.createResultDirectory()
        os.mkdir(PATH_RESULT + path)
        self.loggingBefore()
        self.loggingData(PATH_RESULT + path + PATH_RESULT_PNG)
        self.loggingModel()
        self.loggingResult(PATH_RESULT + path + PATH_RESULT_MODEL)
        # TODO file path
        self.write(PATH_RESULT + path + PATH_RESULT_TXT)

    def checkKey(self, key: str) -> None:
        # TODO check key
        version1, _, _, _ = key.split(DELIMITER)
        version2, _, _, _ = LoggingImpl.KEY.split(DELIMITER)

        if version1 == version2:
            return
        else:
            raise ValueError(VERSION_NOT_SUPPORTED)
