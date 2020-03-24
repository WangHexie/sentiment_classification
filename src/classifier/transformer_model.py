import abc
import numpy as np
from ktrain import text
import ktrain


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
                 epochs=3):
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

    def predict(self, data: list):
        prob = np.array(self.predictor.predict_proba(list(data)))[:, 1]
        labels = (prob > self.threshold).astype(int)
        return labels

    def predict_prob(self, data: list):
        return self.predictor.predict_proba(data)

    def set_threshold(self, x, y):
        pov_num = (np.array(y) == 1).sum()
        pov_prediction = np.array(self.predict_proba(list(x))[:, 1])
        self.threshold = np.sort(pov_prediction)[::-1][pov_num:pov_num + 2].mean()
        return self

    def train(self, x, y):
        # only support binary classification
        full_length = len(y)
        pov_num = (np.array(y) == 1).sum()
        neg_num = full_length - pov_num

        t = text.Transformer(self.model_name, maxlen=self.max_len, class_names=["0", "1"])
        trn = t.preprocess_train(x, y.to_list())
        val = t.preprocess_test(x[:100], y.to_list()[:100])

        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=self.batch_size)
        learner.fit_onecycle(self.learning_rate, self.epochs, class_weight={0: pov_num, 1: neg_num})

        self.learner = learner
        self.predictor = ktrain.get_predictor(learner.model, t)
        self.set_threshold(x, y)

        return self

