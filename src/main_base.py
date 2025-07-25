import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.

project_dir = "/mnt/c/SKH/ai_lab_13/projects/nlp-text-summarization/song"

import sys
sys.path.append(
    project_dir
)
from src.dataset.dataset_base import *
from src.dataset.preprocess import *
from src.models.BART import *
from src.trainer.trainer_base import *

def main(config):
    try:
        # 사용할 device를 정의합니다.
        device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
        print('-'*10, f'device : {device}', '-'*10,)
        print(torch.__version__)

        # 사용할 모델과 tokenizer를 불러옵니다.
        generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
        print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

        # 학습에 사용할 데이터셋을 불러옵니다.
        preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token']) # decoder_start_token: str, eos_token: str
        data_path = config['general']['data_path']
        train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)

        # Trainer 클래스를 불러옵니다.
        trainer = load_trainer_for_train(config, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
        trainer.train()   # 모델 학습을 시작합니다.
    finally:
        # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
        wandb.finish()

if __name__ == "__main__":
    os.chdir(project_dir) ## project directory로 현재 실행 위치를 옮긴다.

    # Load Configuration file
    config_path = os.path.join(
        project_dir,'src','configs',
        ".yaml" # config 파일 이름을 설정
    )
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    main(loaded_config)