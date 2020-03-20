import os

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.config.configs import FilePath
from src.data.basic_functions import root_dir


class Dataset:
    @staticmethod
    def read_original_label(label_path=FilePath.label_path):
        return pd.read_csv(os.path.join(root_dir(), *label_path), header=None, index_col=False,
                           na_values="Nan", keep_default_na=False, encoding="utf-8")

    @staticmethod
    def read_original_data(data_path=FilePath.data_path):
        return pd.read_csv(os.path.join(root_dir(), *data_path), names=["data"], index_col=False,
                           encoding="utf-8")

    @staticmethod
    def _transform_original_label_to_one_hot(labels):
        enc = MultiLabelBinarizer()
        new = enc.fit_transform(labels)
        return pd.DataFrame(new, columns=enc.classes_)

    @staticmethod
    def read_one_hot_label(label_path=FilePath.label_path):
        return Dataset._transform_original_label_to_one_hot(Dataset.read_original_label(label_path).values)

    @staticmethod
    def get_train_file(data: pd.DataFrame, one_hot_label: pd.DataFrame):
        features = one_hot_label.columns
        labels = [(one_hot_label[feature] == 1).map(lambda x:int(x)) for feature in features]
        return [data.copy().assign(label=label) for label in labels], features

    @staticmethod
    def save_splitted_file(data: pd.DataFrame, one_hot_label: pd.DataFrame, dir_name=''):
        df, label_name = Dataset.get_train_file(data, one_hot_label)
        for i in range(len(label_name)):
            df[i].to_csv(os.path.join(root_dir(), "data", dir_name, label_name[i]+".csv"))


if __name__ == '__main__':
    print(Dataset.save_splitted_file(Dataset.read_original_data(FilePath.validation_data_path), Dataset.read_one_hot_label(FilePath.validation_label_path), dir_name='test'))
