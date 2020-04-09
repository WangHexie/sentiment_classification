import abc
import os

import numpy as np
from ktrain import text
import ktrain
import pandas as pd
from sklearn.model_selection import train_test_split

class Classifier:
    @abc.abstractmethod
    def predict(self, data):
        pass

    @abc.abstractmethod
    def predict_prob(self, data):
        pass

    @abc.abstractmethod
    def train(self, x, y):
        pass


class TransformerClassifier(Classifier):
    def __init__(self, model_name='clue/albert_chinese_small', max_len=50, batch_size=64, learning_rate=3e-5,
                 epochs=3,  early_stopping=3, reduce_on_plateau=1):
        self.threshold = None
        self.model = None
        self.learner = None
        self.predictor = None
        self.threshold = None

        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.reduce_on_plateau = reduce_on_plateau

    def predict(self, data: list):
        prob = np.array(self.predict_prob(list(data)))
        return self.prob_convert_to_label(prob)

    def prob_convert_to_label(self, prob):
        prob = np.array(prob)
        labels = (prob > self.threshold).astype(int)
        return labels

    def predict_prob(self, data: list):
        # TODO: bug warning! output of different model may have different shape
        return np.array(self.predictor.predict_proba(data))[0][:, 1]

    def set_threshold(self, x, y):
        pov_num = (np.array(y) == 1).sum()
        pov_prediction = np.array(self.predict_prob(list(x)))
        self.threshold = np.sort(pov_prediction)[::-1][pov_num:pov_num + 2].mean()
        print("threshold:", self.threshold)
        return self

    def save_prob_prediction_result(self, prob, label_name, save_path):
        pd.DataFrame(prob).to_csv(os.path.join(save_path, label_name+"prob"+str(self.threshold)), encoding="utf-8")

    def train(self, x, y):
        # only support binary classification
        full_length = len(y)
        pov_num = (np.array(y) == 1).sum()
        neg_num = full_length - pov_num

        t = text.Transformer(self.model_name, maxlen=self.max_len, class_names=["0", "1"])
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        trn = t.preprocess_train(train_x, train_y.to_list())
        val = t.preprocess_test(test_x, test_y.to_list())

        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=self.batch_size)
        # TODO: Done ==========disable class_weight
        # TODO: =============== add early top parameter into config
        learner.autofit(self.learning_rate, self.epochs, early_stopping=self.early_stopping, reduce_on_plateau=self.reduce_on_plateau)

        self.learner = learner
        self.predictor = ktrain.get_predictor(learner.model, t)
        # TODO: ====================lower number of x
        print("use part of train data")
        x, _, y, _ = train_test_split(x, y, test_size=0.3)  # TODO: hard-code size value
        self.set_threshold(x, y)

        return self

