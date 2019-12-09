# 作者 : lxw
# 文件 : loggingImpl.py
# 日期 : 2019/12/9 下午2:47
# IDE : PyCharm
# Description : loggings的实现类
# Github : https://github.com/liuxw123

from loggings import ResultLog
from values.strings import DELIMITER, VERSION_NOT_SUPPORTED


class LoggingImpl(ResultLog):
    """
    TODO input right key

    """
    KEY = "v0-3-0-0"

    def logging(self):
        pass

    def checkKey(self, key: str) -> None:
        # TODO check key
        version1, sub1, modelVersion1, dataVersion1 = key.split(DELIMITER)
        version2, sub2, modelVersion2, dataVersion2 = LoggingImpl.KEY.split(DELIMITER)

        if version1 == version2 and sub1 == sub2 and modelVersion1 == modelVersion2 and dataVersion1 == dataVersion2:
            return
        else:
            raise ValueError(VERSION_NOT_SUPPORTED)
