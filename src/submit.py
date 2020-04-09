from dataclasses import asdict

from src.classifier.transformer_model import TransformerClassifier
import os

from src.config.configs import FilePath, ClassifierParam
from src.data.basic_functions import root_dir
from src.data.dataset import Dataset
import pandas as pd


def get_all_labels(path=None):
    if path is None:
        labels = Dataset.read_original_label()
    else:
        labels = Dataset.read_original_label((path, FilePath.label_path[1]), mode="test")
    return set(labels.values.flatten().tolist()) - {''}


def save_result_in_dir(final, save_path):
    pd.DataFrame(final).to_csv(save_path, header=False, index=False, encoding="utf-8")


def prediction_all(test_dir=FilePath.validation_data_path[0],
                   train_file_dir=FilePath.data_path[0],
                   prediction_file_path=None,
                   config=None,
                   temp_path=None):
    labels = get_all_labels(train_file_dir)
    test_text = Dataset.read_original_data((test_dir, FilePath.data_path[1]), mode="test")

    Dataset.save_splitted_file(Dataset.read_original_data((train_file_dir, FilePath.data_path[1]), mode="test"),
                               Dataset.read_one_hot_label((train_file_dir, FilePath.label_path[1]), mode="test"),
                               dir_path=train_file_dir)

    if temp_path is None:
        temp_path = root_dir()

    for label in labels:
        print(label)
        train_data = Dataset.read_splitted_train_file(os.path.join(train_file_dir, label+".csv"))
        cls = TransformerClassifier(**config).train(train_data["data"], train_data["label"])

        prob = cls.predict_prob(test_text["data"].values.tolist())
        cls.save_prob_prediction_result(prob, label, temp_path)

        label_prediction_result = cls.prob_convert_to_label(prob)
        Dataset.save_label_prediction_result(label_prediction_result, label, temp_path)

    final = Dataset.merge_all_prediction_in_dir(temp_path, labels)
    save_result_in_dir(final, prediction_file_path)
    # read train file by labels
    #

if __name__ == '__main__':
    prediction_all(test_dir=os.path.join(root_dir(), *FilePath.validation_data_path[:2]),
                   train_file_dir=os.path.join(root_dir(), FilePath.data_path[0]),
                   prediction_file_path=None,
                   config=asdict(ClassifierParam(epochs=4, batch_size=1, max_len=10, model_name='clue/roberta_chinese_base')))
    # print(Dataset.read_original_data((os.path.join(root_dir(), "data"), FilePath.data_path[1]) ,mode="test"))