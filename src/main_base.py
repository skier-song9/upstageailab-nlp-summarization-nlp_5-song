import argparse
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

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.

project_dir = "/data/ephemeral/home/nlp-5/auto1p"

import sys
sys.path.append(
    project_dir
)
from src.dataset.dataset_base import *
from src.dataset.preprocess import *
# from src.models.BART import *
from src.models.AutoModels import *
from src.trainer.trainer_base import *
from src.inference.inference import *

def main(config, practice=False):
    try:
        pl.seed_everything(seed=config['training']['seed'], workers=False) # workers : worker 프로세스 시드는 고정하지 않음  > 과적합 방지.
        # 사용할 device를 정의합니다.
        device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
        print('-'*10, f'device : {device}', '-'*10,)
        print(torch.__version__)

        # 사용할 모델과 tokenizer를 불러옵니다.
        generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
        print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

        # 학습에 사용할 데이터셋을 불러옵니다.
        summ_train_dataset, summ_val_dataset = prepare_train_dataset(
            tokenizer=tokenizer,
            config=config,
            practice=practice
        )

        # Trainer 클래스를 불러옵니다.
        trainer = load_trainer_for_train(
            config=config,
            generate_model=generate_model,
            tokenizer=tokenizer,
            train_inputs_dataset=summ_train_dataset,
            val_inputs_dataset=summ_val_dataset
        )
        print()
        print("--- Start train ---")
        trainer.train()   # 모델 학습을 시작합니다.
        print("--- Finish train ---")
        print()

        # best 모델과 토크나이저 저장
        trainer.model.save_pretrained(config['inference']['ckt_dir'])
        tokenizer.save_pretrained(config['inference']['ckt_dir'])

        # validation 후 val_inference.csv 파일 저장.
        val_infer_df, summ_val_infer_dataset = prepare_test_dataset(
            config=config,
            tokenizer=tokenizer,
            val_flag=True,
            practice=practice
        )
        print()
        print("--- Start Validation inference ---")
        _ = inference(
            config=config,
            generate_model=trainer.model,
            tokenizer=tokenizer,
            test_df=val_infer_df,
            summ_test_dataset=summ_val_infer_dataset,
            val_flag=True
        )
        print("--- Finish Validation inference ---")
        print()

        # inference 후 submission 파일 저장.
        test_df, summ_test_dataset = prepare_test_dataset(
            config=config,
            tokenizer=tokenizer,
            val_flag=False,
            practice=practice
        )
        print()
        print("--- Start Test inference ---")
        _ = inference(
            config=config,
            generate_model=trainer.model,
            tokenizer=tokenizer,
            test_df=test_df,
            summ_test_dataset=summ_test_dataset,
            val_flag=False
        )
        print("--- Finish Test inference ---")
        print()
    finally:
        # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
        wandb.finish()

if __name__ == "__main__":
    os.chdir(project_dir) ## project directory로 현재 실행 위치를 옮긴다.

    parser = argparse.ArgumentParser(description="Run deep learning training with specified configuration.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml', # 기본값 설정
        help='Name of the configuration YAML file (e.g., config.yaml, experiment_A.yaml)'
    )
    parser.add_argument(
        '--inference',
        type=bool,
        default=False,
        help='executing this file as inference mode'
    )
    parser.add_argument(
        '--practice',
        type=bool,
        default=False,
        help='Set to True to check if errors occur when testing your code.'
    )
    # True로 설정하면 validation 데이터셋, test 데이터셋을 10개만 사용해 전체 과정이 더 빠르게 진행되도록 한다.

    args = parser.parse_args()

    # Load Configuration file
    config_path = os.path.join(
        project_dir,'src','configs',
        args.config # config 파일 이름을 설정
    )
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    # CUDA device 관련 오류 메세지 보기
    os.environ['TORCH_USE_CUDA_DSA'] = 'true'

    if not args.inference:
        main(loaded_config, args.practice)
    else:
        device = "cuda:0" if torch.cuda.is_available else "cpu"
        generate_model, tokenizer = load_tokenizer_and_model_for_inference(loaded_config, device)
        test_df, summ_test_dataset = prepare_test_dataset(
            config=loaded_config,
            tokenizer=tokenizer,
            val_flag=False
        )
        _ = inference(
            config=loaded_config,
            generate_model=generate_model,
            tokenizer=tokenizer,
            test_df=test_df,
            summ_test_dataset=summ_test_dataset,
            val_flag=False
        )

