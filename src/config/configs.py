from dataclasses import dataclass


@dataclass
class FilePath:
    label_path: tuple = ("data", "labels.csv")
    data_path: tuple = ("data", "texts.csv")
    validation_data_path: tuple = ("data", "test", "texts.csv")
    validation_label_path: tuple = ("data", "test", "labels.csv")
