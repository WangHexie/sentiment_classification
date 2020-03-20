import abc
import numpy as np


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
    def __init__(self):
        self.threshold = None
        self.model = None
        pass

    def predict(self, data):
        pass

    def predict_prob(self, data):
        pass

    def train(self, x, y):
        # only support binary classification
        class_num = dict([(i, y.count(i)) for i in set(y)])
        class_weight = dict([[0, class_num[1]], [1, class_num[0]]])  # bug

        prediction_on_train_data = self.predict_prob(x)
        threshold = np.array(prediction_on_train_data).sort()[class_num[0]:class_num[0]+2].mean()

        self.threshold = threshold   # set threshold
        pass

