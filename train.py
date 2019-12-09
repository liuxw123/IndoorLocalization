# 作者 : lxw
# 文件 : train.py
# 日期 : 2019/12/4 18:33
# IDE : PyCharm
# Description : training file
# Github : https://github.com/liuxw123

from modelConfig import KEY
from modelDefinitionImpl import PstModelV0
from trainData import TrainData, collate
from loggingImpl import LoggingImpl

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch

# parameters
TRAIN_RATE = 0.9
BATCH_SIZE = 32
LR = 0.001
EPOCH = 200
ADJUST_EPOCH = [30, 70]

dataSet = TrainData(TRAIN_RATE, KEY)  # default is train phase
dataLoader = DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
lossFunc = CrossEntropyLoss()

model = PstModelV0(KEY)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=0.0001)

logger = LoggingImpl(KEY)


def adjustLearningRate(epoch):
    if epoch not in ADJUST_EPOCH:
        return

    idx = ADJUST_EPOCH.index(epoch) + 1
    lr = LR / (pow(10, idx))

    for para in optimizer.param_groups:
        para['lr'] = lr


def logging():
    info1 = dataSet.dataHolder.details()
    info2 = model.details()

    info3 = {"batch": BATCH_SIZE, "optimizer": type(optimizer).__name__, "lr": LR, "epoch": EPOCH,
             "loss function": type(lossFunc).__name__}

    logger.add("Data", info1)
    logger.add("Model", {**info2, **info3})


def save():
    pass


def test():
    dataSet.changeToTestPhase()
    model.eval()
    nSum = 0
    nCor = 0
    for (x, y) in dataLoader:
        out = model(x)
        predict = out.max(dim=1)[1]
        nSum += y.shape[0]
        nCor += torch.sum(predict == y)

    return nCor.item() / nSum


def train(epoch) -> float:
    dataSet.changeToTrainPhase()
    model.train()

    losses = 0
    for cnt, (x, y) in enumerate(dataLoader):
        optimizer.zero_grad()
        out = model(x)
        loss = lossFunc(out, y)
        loss.backward()
        optimizer.step()

        losses += loss.item()

    adjustLearningRate(epoch)
    return losses / (cnt + 1)


def main(numEpoch):
    for epoch in range(numEpoch):
        loss = train(epoch)
        accuracy = test()

        print("Epoch: {:>3d} loss: {:.4f} test accuracy: {:.2f}%".format(epoch, loss, accuracy * 100))


if __name__ == '__main__':
    logging()
    main(EPOCH)
