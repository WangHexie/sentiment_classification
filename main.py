import argparse
from dataclasses import asdict

from src.config.configs import ClassifierParam
from src.submit import prediction_all, classifier_ensemble

parser = argparse.ArgumentParser("sentiment classification", fromfile_prefix_chars='@')
parser.add_argument('--traning-dataset', type=str, help='train_dataset input path')
parser.add_argument('--prediction-file', type=str, help='prediction output path')
parser.add_argument('--test-dataset', type=str, help='test-dataset input path')
parser.add_argument('--model', type=str, default='hfl/chinese-bert-wwm-ext', help='model name')

parser.add_argument('--batch_size', type=int, default=196, help='batch_size')
parser.add_argument('--max_len', type=int, default=50, help='max_len')
parser.add_argument('--temp_path', type=str, default="./", help='temp length')

parser.add_argument('--early_stopping', type=int, default=3, help='early_stopping')
parser.add_argument('--reduce_on_plateau', type=int, default=1, help='reduce_on_plateau')

args = parser.parse_args()

prediction_paths = ["./test1.csv", "./test2.csv"]

prediction_all(test_dir=args.test_dataset,
               train_file_dir=args.traning_dataset,
               prediction_file_path=prediction_paths[0],
               config=asdict(
                   ClassifierParam(epochs=30, batch_size=args.batch_size, max_len=args.max_len, model_name=args.model,
                                   learning_rate=1e-5, early_stopping=args.early_stopping,
                                   reduce_on_plateau=args.reduce_on_plateau)),
               temp_path=args.temp_path)

prediction_all(test_dir=args.test_dataset,
               train_file_dir=args.traning_dataset,
               prediction_file_path=prediction_paths[1],
               config=asdict(
                   ClassifierParam(epochs=30, batch_size=156, max_len=85, model_name=args.model,
                                   learning_rate=1e-5, early_stopping=args.early_stopping,
                                   reduce_on_plateau=args.reduce_on_plateau)),
               temp_path=args.temp_path)


classifier_ensemble(prediction_paths, output_path=args.prediction_file)

