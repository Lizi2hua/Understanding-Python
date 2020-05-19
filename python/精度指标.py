import matplotlib.pyplot as plt
# plot(​x, y​)
import numpy01 as np

#需要改
# python list除法不支持，用np
class precision_index(object):
    def __init__(self, TP, FP, TN, FN):
        # cofusion matrix
        # true positive
        self.TP = np.array(TP)
        # false positive
        self.FP = np.array(FP)
        # true negative
        self.TN = np.array(TN)
        # false negative
        self.FN = np.array(FN)
        # true positive rate
        self.TPR = np.array(TP) / (np.array(TP) + np.array(FN))
        # false positive rate
        self.FPR = np.array(FP) / (np.array(FP) + np.array(TN))
        # true negative rate
        self.TNR = np.array(TN) / (np.array(TN) + np.array(FP))
        # false negative rate
        self.FNR = np.array(FN) / (np.array(FN) + np.array(TP))

    # accuarcy
    def acc(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN
        self.accuarcy = (TP + TN) / (TP + TN + FP + FN)
        return self.accuarcy

    # precision
    def pre(self):
        TP = self.TP
        FP = self.FP
        self.precision = TP / (TP + FP)
        return self.precision

    # recall
    def recall(self):
        TP = self.TP
        FN = self.FN
        self.recall = TP / (TP + FN)
        return self.recall

    #  F1 SCORE
    def f1Score(self):
        pre = self.pre()
        recall = self.recall()
        f1Score = (2 * pre * recall) / (pre + recall)
        return f1Score

    def showPRC(self):
        pre = self.pre()
        recall = self.recall()
        plt.plot(pre, recall)

    def showROC(self):
        fpr = self.FPR
        tpr = self.TPR
        plt.plot(fpr, tpr)


# TEST DATA
# shit!
# 构建测试集，在100个test样本中，80为正，20为负
testTP = [80,75,79,77,80]
testFP = [1,2,3,2,0]
testTN = [19,18,17,18,0]
testFN = [0,5,1,3,0]
index = precision_index(TP=testTP, FP=testFP, TN=testTN, FN=testFN)
# index.showPRC()
index.showROC()
