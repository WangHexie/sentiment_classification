import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.config.configs import FilePath
from src.data.basic_functions import root_dir


class Dataset:
    @staticmethod
    def read_original_label(label_path=FilePath.label_path, mode="train"):
        if mode == "train":
            return pd.read_csv(os.path.join(root_dir(), *label_path), header=None, index_col=False,
                            na_values="Nan", keep_default_na=False, encoding="utf-8")
        else:
            return pd.read_csv(os.path.join(*label_path), header=None, index_col=False,
                               na_values="Nan", keep_default_na=False, encoding="utf-8")

    @staticmethod
    def read_original_data(data_path=FilePath.data_path, mode="train"):
        if mode == "train":
            return pd.read_csv(os.path.join(root_dir(), *data_path), names=["data"], index_col=False,
                               encoding="utf-8")
        else:
            return pd.read_csv(os.path.join(*data_path), names=["data"], index_col=False,
                               encoding="utf-8")

    @staticmethod
    def _transform_original_label_to_one_hot(labels):
        enc = MultiLabelBinarizer()
        new = enc.fit_transform(labels)
        return pd.DataFrame(new, columns=enc.classes_)

    @staticmethod
    def reverse_transform_one_hot_to_label(df_prediction):
        return [np.array(df_prediction.columns)[i].tolist() for i in df_prediction.values]

    @staticmethod
    def read_one_hot_label(label_path=FilePath.label_path, mode="train"):
        return Dataset._transform_original_label_to_one_hot(Dataset.read_original_label(label_path, mode=mode).values)

    @staticmethod
    def read_splitted_train_file(path):
        return pd.read_csv(path, encoding="utf-8")

    @staticmethod
    def get_train_file(data: pd.DataFrame, one_hot_label: pd.DataFrame):
        features = one_hot_label.columns
        labels = [(one_hot_label[feature] == 1).map(lambda x:int(x)) for feature in features]
        return [data.copy().assign(label=label) for label in labels], features

    @staticmethod
    def save_splitted_file(data: pd.DataFrame, one_hot_label: pd.DataFrame, dir_path=''):
        df, label_name = Dataset.get_train_file(data, one_hot_label)
        for i in range(len(label_name)):
            df[i].to_csv(os.path.join(dir_path, label_name[i]+".csv"), encoding="utf-8")

    @staticmethod
    def save_label_prediction_result(label_prediction_result, label_name, save_path):
        pd.DataFrame(label_prediction_result).to_csv(os.path.join(save_path, label_name), encoding="utf-8")

    @staticmethod
    def merge_all_prediction_in_dir(save_dir, label_names):
        final_result = []
        for label in label_names:
            data = pd.read_csv(os.path.join(save_dir, label))
            temp = []

            prediction = data.iloc[:, 1].values.tolist()
            for i in prediction:
                if i == 1:
                    temp.append([label])
                else:
                    temp.append([])

            final_result.append(temp)

        result = []
        for i in range(len(data)):
            temp = []
            for k in range(len(label_names)):
                temp = temp + final_result[k][i]
            result.append(temp)  # TODO: ============= bug inspect

        return result

    @staticmethod
    def save_result_in_dir(final, save_dir):
        pass



if __name__ == '__main__':
    print(Dataset.save_splitted_file(Dataset.read_original_data(FilePath.validation_data_path), Dataset.read_one_hot_label(FilePath.validation_label_path), dir_path=os.path.join(root_dir(), "data", 'test')))
