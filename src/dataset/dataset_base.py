import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import Dict
from transformers import AutoTokenizer

class SummDataset(Dataset):
    """pd.DataFrame을 torch.utils.data.Dataset으로 변환하는 클래스"""
    def __init__(self, tokenized_data, tokenizer, config):
        """
        :param Dict tokenized_data: tokenizer.tokenize가 완료된 딕셔너리 데이터.
        :param transformers.AutoTokenizer tokenizer: tokenizer
        :param Dict config: 혹시 모를 추가 기능에 대비한 config 인자.
        """
        self.tokenized_data = tokenized_data
        self.tokenizer = tokenizer
        self.config = config
    def __getitem__(self, index):
        input_ids = torch.tensor(self.tokenized_data['input_ids'][index])

        # 추론용 데이터셋인 경우 {"input_ids":[[tokens]...], "labels": None} 임.
        labels = self.tokenized_data['labels']
        if labels is not None:
            labels = torch.tensor(labels[index])

        # attention_mask를 생성
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    def __len__(self):
        return len(self.tokenized_data['input_ids'])

def tokenize_data(df:pd.DataFrame, tokenizer:AutoTokenizer, config:Dict, test:bool=False):
    """pd.DataFrame에서 dialogue와 summary를 토큰화하는 함수

    :param pd.DataFrame df: train, dev, test csv
    :param transformers.AutoTokenizer tokenizer: tokenizer
    :param Dict config: _description_
    :param bool test: True이면 summary를 토큰화하지 않는다.
    :return _type_: _description_
    """
    dialogues = df['dialogue']
    # tokenize dialogue
    tokenized_dialogues = [
        tokenizer(
            dialogue,
            padding='max_length',
            truncation=True,
            max_length=config['tokenizer']['encoder_max_len'],
            add_special_tokens=True
        )['input_ids'] for dialogue in dialogues.values
    ]
    
    # summary 처리 
    # test의 경우 summary가 없으니, None으로 출력.
    # train의 경우 summary가 있으니, summary를 토큰화하여 labels를 채운다. 
    tokenized_summaries = None
    if not test:
        summaries = df['summary']
        tokenized_summaries = [
            tokenizer(
                summary,
                padding='max_length',
                truncation=True,
                max_length=config['tokenizer']['decoder_max_len'],
                add_special_tokens=True
            )['input_ids'] for summary in summaries.values
        ]
        # 패딩된 부분을 -100으로 치환하여 학습에서 제외합니다.
        tokenized_summaries = [[-100 if token == tokenizer.pad_token_id else token for token in summary] for summary in tokenized_summaries]

    out = {'input_ids': tokenized_dialogues, 'labels': tokenized_summaries}
    print("="*15, "데이터 개수" ,"="*15)
    print("tokenizing 된 데이터 형태 예시")
    print(tokenizer.convert_ids_to_tokens(tokenized_dialogues[-1]))
    print("label의 형태 예시")
    print(tokenized_summaries[-1])
    print("="*15, "데이터 개수" ,"="*15)
    return out
    
def prepare_train_dataset(tokenizer, config):
    """train, val, test SummDataset을 준비

    :param transformers.AutoTokenizer tokenizer: tokenizer
    :param Dict config: _description_
    :return _type_: _description_
    """
    # load data
    train_df = pd.read_csv(os.path.join(config['general']['data_path'], config['general']['train_data']))
    val_df = pd.read_csv(os.path.join(config['general']['data_path']['val_data']))
    test_df = pd.read_csv(os.path.join(config['general']['data_path'], config['general']['test_data']))


    # print data info
    print("="*15, "데이터 개수" ,"="*15)
    print(f"train_df.shape: {train_df.shape}")
    print(f"val_df.shape: {val_df.shape}")
    print(f"test_df.shape: {test_df.shape}")
    print("="*15, "데이터 개수" ,"="*15)
    print()

    # tokenize
    print("="*15, "토큰화 진행 중..." ,"="*15)
    tokenized_train = tokenize_data(df=train_df, tokenizer=tokenizer, config=config, test=False)
    tokenized_val = tokenize_data(df=val_df, tokenizer=tokenizer, config=config, test=False)
    # tokenized_test = tokenize_data(df=test_df, tokenizer=tokenizer, config=config, test=True)
    print("="*15, "토큰화 완료" ,"="*15)
    print()

    # make SummDataset
    print("="*15, "make SummDataset..." ,"="*15)
    summ_train_dataset = SummDataset(tokenized_data=tokenized_train, tokenizer=tokenizer, config=config)
    summ_val_dataset = SummDataset(tokenized_data=tokenized_val, tokenizer=tokenizer, config=config)
    # summ_test_dataset = SummDataset(tokenized_data=tokenized_test, tokenizer=tokenizer, config=config)
    print("="*15, "SummDataset 완료" ,"="*15)

    return summ_train_dataset, summ_val_dataset

def prepare_test_dataset(config, tokenizer, val_flag=False):

    if val_flag:
        test_file_path = os.path.join(config['general']['data_path']['val_data'])
    else:
        test_file_path = os.path.join(config['general']['data_path']['test_data'])

    test_df = pd.read_csv(test_file_path)

    print('-'*150)
    print(f'test_data:\n{test_df["dialogue"][0]}')
    print('-'*150)

    tokenized_test = tokenize_data(df=test_df, tokenizer=tokenizer, config=config, test=True)
    summ_test_dataset = SummDataset(tokenized_data=tokenized_test, tokenizer=tokenizer, config=config)

    return test_df, summ_test_dataset