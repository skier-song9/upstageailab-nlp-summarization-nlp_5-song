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
        input_ids = self.tokenized_data['input_ids'][index]

        # 추론용 데이터셋인 경우 {"input_ids":[[tokens]...], "labels": None} 임.
        labels = self.tokenized_data['labels'] ### 문제의 코드
        if labels is not None:
            labels = labels[index]
        else: # None 인 경우
            labels = None

        # attention_mask를 생성 > attention_mask는 DataCollator가 자동으로 생성.
        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels)
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
            # padding=False, # DataCollatorForSeq2Seq에서 동적으로 padding을 하게 된다. 따라서 여기서는 padding을 하지 않는다.
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
                # padding=False, # DataCollatorForSeq2Seq에서 동적으로 padding을 하게 된다. 따라서 여기서는 padding을 하지 않는다.
                truncation=True,
                max_length=config['tokenizer']['decoder_max_len'],
                add_special_tokens=True
            )['input_ids'] for summary in summaries.values
        ]
        # 패딩된 부분을 -100으로 치환하여 학습에서 제외합니다.
        """
        
        """
        # tokenized_summaries = [[-100 if token == tokenizer.pad_token_id else token for token in summary] for summary in tokenized_summaries]

    out = {'input_ids': tokenized_dialogues, 'labels': tokenized_summaries}
    print("="*15, "tokenizing start" ,"="*15)
    print("tokenizing 된 데이터 형태 예시")
    print(tokenizer.convert_ids_to_tokens(tokenized_dialogues[-1]))
    print("label의 형태 예시")
    print(tokenizer.convert_ids_to_tokens(tokenized_summaries[-1]) if tokenized_summaries is not None else "None")
    print("="*15, "tokenizing end" ,"="*15)
    return out
    
def prepare_train_dataset(tokenizer, config, practice=False):
    """train, val, test SummDataset을 준비

    :param transformers.AutoTokenizer tokenizer: tokenizer
    :param Dict config: _description_
    :param bool practice: True이면, 코드 실험용이므로 train은 256, val은 10개만 반환한다.
    :return _type_: _description_
    """
    # load data
    print("="*15, "load train" ,"="*15)
    train_df = Preprocess.make_set_as_df(
        file_path=config['general']['train_data'],
        is_train=True,
        config=config
    )
    print("="*15, "load val" ,"="*15)
    val_df = Preprocess.make_set_as_df(
        file_path=config['general']['val_data'],
        is_train=True,
        config=config
    )
    print("="*15, "load test" ,"="*15)
    test_df = Preprocess.make_set_as_df(
        file_path=config['general']['test_data'],
        is_train=False,
        config=config
    )

    if practice:
        train_df = train_df.iloc[:256]
        val_df = val_df.iloc[:10]
        test_df = test_df.iloc[:10]

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

    print("="*15, "SummDataset 확인" ,"="*15)
    out = summ_train_dataset.__getitem__(0)
    print("="*15, "SummDataset 확인 완료" ,"="*15)

    return summ_train_dataset, summ_val_dataset

def prepare_test_dataset(config, tokenizer, val_flag=False, practice=False):

    if val_flag:
        test_file_path = os.path.join(config['general']['data_path'], config['general']['val_data'])
    else:
        test_file_path = os.path.join(config['general']['data_path'], config['general']['test_data'])

    test_df = pd.read_csv(test_file_path)

    if practice:
        test_df = test_df.iloc[:10]

    print('-'*150)
    print(f'test_data:\n{test_df["dialogue"][0]}')
    print('-'*150)

    tokenized_test = tokenize_data(df=test_df, tokenizer=tokenizer, config=config, test=True)
    summ_test_dataset = SummDataset(tokenized_data=tokenized_test, tokenizer=tokenizer, config=config)

    return test_df, summ_test_dataset


### 데이터 전처리 함수 ###
import re
from typing import List

# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환
class Preprocess:
    # 클래스 초기화 메서드
    def __init__(self) -> None:
        pass

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    # 정적 메서드로, 클래스 인스턴스 없이 호출 가능
    def make_set_as_df(file_path, is_train = True, config=None):
        def load_df(file_path, is_train, config):
            df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
            # 🔁 발화자 기반 지시표현 보완 전처리 적용
            df['dialogue'] = df['dialogue'].apply(resolve_deictic_with_speaker)
            # 🔁 텍스트 클린 함수
            df['dialogue'] = df['dialogue'].apply(clean_text)

            ### special token에 #Topic# 이 있으면, 지시어 프롬프트에 추가.
            if config is not None and '#Topic#' in config['tokenizer']['special_tokens']:
                df = df.apply(add_instructions, axis=1)

            ### change \n to SEP token
            if config is not None and config['tokenizer'].get('sep_token', None):
                df['dialogue'] = df['dialogue'].apply(lambda x : apply_sep_token(x, config['tokenizer']['sep_token']))

            # is_train 플래그가 True이면 학습용 데이터로 처리
            if is_train:
                train_df = df[['fname','dialogue','summary']] # 'fname', 'dialogue', 'summary' 컬럼 선택
                return train_df # 생성된 학습 데이터프레임 반환
            # is_train 플래그가 False이면 테스트용 데이터로 처리
            else:
                test_df = df[['fname','dialogue']] # 'fname', 'dialogue' 컬럼 선택
                return test_df # 생성된 테스트 데이터프레임 반환

        # 만약 file_path가 리스트로 전달된다면 merge 해라.
        if isinstance(file_path, List):
            df = []
            for fp in file_path:
                df_ = load_df(os.path.join(config['general']['data_path'],fp), is_train, config)
                df.append(df_)
            df = pd.concat(df, axis=0) # 행을 늘림

        else: # file_path가 단일 문자열일 때
            df = load_df(os.path.join(config['general']['data_path'],file_path), is_train, config)

        return df


# 지시표현 보완 함수: 직전 발화자 정보로 지시어 대체
def resolve_deictic_with_speaker(dialogue: str) -> str:
    deictic_phrases = ['그 사람', '이 사람', '그거', '이거', '그건', '이건', '거기', '저기', '여기']
    lines = str(dialogue).split('\n')
    resolved = []
    last_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(#Person\d+#):\s*(.*)', line)
        if match:
            speaker = match.group(1)
            utterance = match.group(2)

            for deictic in deictic_phrases:
                if deictic in utterance and last_speaker:
                    utterance = utterance.replace(deictic, f'{last_speaker}가 말한')

            last_speaker = speaker
            resolved.append(f"{speaker}: {utterance}")
        else:
            resolved.append(line)

    return '\n'.join(resolved)

# 텍스트 클린 함수
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 줄바꿈 표현 통일
    text = text.replace("\\n", "\n").replace("<br>", "\n").replace("</s>", "\n")

    ### 특이 케이스 : train.csv에는 'ㅎㅎ'가 오직 1개 존재한다. 그런데 이것이 #Person2#: ㅎㅎ 라서 빈문자열로 대체하면 말이 없어진다.
    # 문맥과 summary에 맞춰 '나도 행복해.'로 바꾼다.
    text = text.replace("ㅎㅎ", "나도 행복해.")

    # 자소만 있는 단어 제거 (예: ㅋㅋ, ㅇㅋ, ㅜㅜ) > 이모티콘
    text = re.sub(r"\b[ㄱ-ㅎㅏ-ㅣ]{2,}\b", "", text)

    # 중복 줄바꿈 제거
    text = re.sub(r"\n+", r"\n", text)

    # 중복 공백 제거
    text = re.sub(r"\s+", ' ', text)

    return text.strip()

def add_instructions(row:pd.Series) -> pd.Series:
    """지시어 프롬프트 추가.

    :param str dialogue: _description_
    :return str: _description_
    """
    try:
        topic = str(row['topic']).strip()
        dialogue = row['dialogue']
        dialogue = f"#Topic#{topic}\n#Dialogue#{dialogue}"
        row['dialogue'] = dialogue
    ##Topic#','#Dialogue#'
    except:
        return row
    return row

def apply_sep_token(text:str, sep_token:str) -> str:
    return re.sub(r"\n", sep_token, text)