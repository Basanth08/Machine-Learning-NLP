import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True])/len(correct)
        self.confusion_matrix={ }
        #for each class variable,we are storing the tp, fp, fn, and tn values in the values and then in the confusion matrix
        for c in self.classes_:
            tp = np.sum((self.predictions==c) & (self.actuals==c))
            fp = np.sum((self.predictions==c) & (self.actuals!=c)) 
            fn = np.sum((self.predictions!=c) & (self.actuals==c))
            tn = np.sum((self.predictions!=c) & (self.actuals!=c))
            self.confusion_matrix[c] = {"TP":tp, "TN": tn, "FP": fp, "FN": fn}
        return
    
    def accuracy(self):
        if self.confusion_matrix==None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        if self.confusion_matrix==None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp+fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += prec_label * ratio
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        
        if target:
            tp=self.confusion_matrix[target]["TP"]
            fn=self.confusion_matrix[target]["FN"]
            rec=tp/(tp+fn)# formula to calculate recall that is the correct indentificaiton of the positive values from the postiive and negative valeuys

        else:
            rec_list=[]
            for c in self.confusion_matrix:# same as precision
                tp=self.confusion_matrix[c]["TP"]
                fn=self.confusion_matrix[c]["FN"]
                if tp+fn>0:
                    rec_list.append(tp/(tp+fn))
            if average=="macro":
                rec=np.mean(rec_list)
            elif average=="micro":
                tp_sum=sum([self.confusion_matrix[c]["TP"] for c in self.confusion_matrix]) 
                fn_sum=sum([self.confusion_matrix[c]["FN"] for c in self.confusion_matrix])
                rec=tp_sum / (tp_sum + fn_sum)
            elif average=="weighted":
                counts=Counter(self.actuals)
                rec_weighted=0
                for c in self.confusion_matrix:
                    rec_c=self.recall(c)
                    rec_weighted+=rec_c*counts[c]# counts of instances * records of each class here for the weighted average

                rec=rec_weighted/len(self.actuals)

        return rec

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        prec =self.precision(target = target, average = average)
        rec =self.recall(target = target, average = average)
        
        if prec+rec>0:
            f1=2.0*prec*rec/(prec+rec)
        else:
            f1=0
        
        return f1
