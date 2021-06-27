# 情感分类（sentiment classification）
> 文本情感分类，多标签, Transformer, SOTA    
***Multi-label classification based on Transformer***      


***2020年“未来杯高校AI挑战赛” 战疫 赛道总决赛第二名***

## Requirements

```sh
pip install -r requirements.txt
git clone https://github.com/WangHexie/ktrain.git
cd ktrain && pip install .
```

## Usage example

```shell script
python ./main.py --traning-dataset ./data --prediction-file "./data/test.csv" --test-dataset ./data/test  --max_len 80 --model hfl/chinese-roberta-wwm-ext --batch_size 196
```

```shell script
usage: sentiment classification [-h] [--traning-dataset TRANING_DATASET]
                                [--prediction-file PREDICTION_FILE]
                                [--test-dataset TEST_DATASET] [--model MODEL]
                                [--batch_size BATCH_SIZE] [--max_len MAX_LEN]
                                [--temp_path TEMP_PATH]
                                [--early_stopping EARLY_STOPPING]
                                [--reduce_on_plateau REDUCE_ON_PLATEAU]

```






