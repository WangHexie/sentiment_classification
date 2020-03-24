from dataclasses import dataclass, asdict


@dataclass
class FilePath:
    label_path: tuple = ("data", "labels.csv")
    data_path: tuple = ("data", "texts.csv")
    validation_data_path: tuple = ("data", "test", "texts.csv")
    validation_label_path: tuple = ("data", "test", "labels.csv")


@dataclass
class ClassifierParam:
    model_name:str = 'clue/albert_chinese_small'
    max_len:int = 50
    batch_size:int = 64
    learning_rate:float = 3e-5
    epochs:int = 3


default_cls_parm = asdict(ClassifierParam())

