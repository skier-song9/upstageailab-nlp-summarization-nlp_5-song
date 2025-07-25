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

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.

project_dir = "upstageailab-nlp-summarization-nlp_5"

import sys
sys.path.append(
    project_dir
)
# from src.dataset.dataset_base import *
# from src.dataset.preprocess import *
# from src.models.BART import *
# from src.trainer.trainer_base import *
# from src.inference.inference import *

import os
from torch.utils.data import Dataset

# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.decoder_input = decoder_input # 토큰화된 디코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item은 'input_ids'와 'attention_mask'를 키로 가짐
        # 디코더 입력에서 해당 인덱스의 데이터를 복사하여 item2 딕셔너리 생성
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2는 'input_ids'와 'attention_mask'를 키로 가짐
        # 모델의 디코더 입력 키 이름('decoder_input_ids')에 맞게 'input_ids'를 변경
        item2['decoder_input_ids'] = item2['input_ids']
        # 모델의 디코더 어텐션 마스크 키 이름('decoder_attention_mask')에 맞게 'attention_mask'를 변경
        item2['decoder_attention_mask'] = item2['attention_mask']
        # 기존의 'input_ids' 키를 item2에서 제거
        item2.pop('input_ids')
        # 기존의 'attention_mask' 키를 item2에서 제거
        item2.pop('attention_mask')
        # item 딕셔너리에 item2 딕셔너리의 내용을 추가
        item.update(item2) # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'를 키로 가짐
        # 최종적으로 'labels' 키에 해당 인덱스의 레이블(토큰화된 디코더 출력)을 추가
        item['labels'] = self.labels['input_ids'][idx] # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'를 키로 가짐
        return item # 모델의 입력으로 사용될 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.decoder_input = decoder_input # 토큰화된 디코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item은 'input_ids'와 'attention_mask'를 키로 가짐
        # 디코더 입력에서 해당 인덱스의 데이터를 복사하여 item2 딕셔너리 생성
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2는 'input_ids'와 'attention_mask'를 키로 가짐
        # 모델의 디코더 입력 키 이름('decoder_input_ids')에 맞게 'input_ids'를 변경
        item2['decoder_input_ids'] = item2['input_ids']
        # 모델의 디코더 어텐션 마스크 키 이름('decoder_attention_mask')에 맞게 'attention_mask'를 변경
        item2['decoder_attention_mask'] = item2['attention_mask']
        # 기존의 'input_ids' 키를 item2에서 제거
        item2.pop('input_ids')
        # 기존의 'attention_mask' 키를 item2에서 제거
        item2.pop('attention_mask')
        # item 딕셔너리에 item2 딕셔너리의 내용을 추가
        item.update(item2) # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'를 키로 가짐
        # 최종적으로 'labels' 키에 해당 인덱스의 레이블(토큰화된 디코더 출력)을 추가
        item['labels'] = self.labels['input_ids'][idx] # 이제 item은 'input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'를 키로 가짐
        return item # 모델의 입력으로 사용될 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# Test에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForInference(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.test_id = test_id # 테스트 데이터의 ID를 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 인코더 입력에서 해당 인덱스의 데이터를 복사하여 item 딕셔너리 생성
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        # item 딕셔너리에 'ID' 키와 해당 인덱스의 테스트 ID 값을 추가
        item['ID'] = self.test_id[idx]
        return item # 모델의 입력 및 ID를 포함하는 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_dir, tokenizer):
    """_summary_

    :param dictionary config: 각종 설정 딕셔너리
    :param Preprocess preprocessor: Preprocess 객체
    :param str data_dir: data 디렉토리 경로
    :param AutoTokenizer tokenizer: tokenizer 객체
    :return torch.utils.data.Dataset: Custom Dataset for train, val
    """
    train_file_path = os.path.join(data_dir,'train.csv')
    val_file_path = os.path.join(data_dir,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print("train dataframe columns:",train_data.columns)
    print("val dataframe columns:",val_data.columns)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')




    # Enc-Dec 구조의 모델인 경우 Encoder input과 Decoder input을 구분.
    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(
        encoder_input_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    tokenized_decoder_inputs = tokenizer(
        decoder_input_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    tokenized_decoder_ouputs = tokenizer(
        decoder_output_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )
    val_tokenized_decoder_ouputs = tokenizer(
        decoder_output_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback
import torch
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
import wandb
import os
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {key: value["f"] for key, value in results.items()}
    return result

# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)

    training_args = Seq2SeqTrainingArguments(
        seed=config['training']["seed"],
        output_dir=config['training']["output_dir"],
        overwrite_output_dir=config['training']["overwrite_output_dir"],

        save_total_limit=config['training']["save_total_limit"],
        load_best_model_at_end=config['training']["load_best_model_at_end"],
        save_steps=config['training']["save_steps"],

        logging_steps=config['training']["logging_steps"],

        num_train_epochs=config['training']["num_train_epochs"],
        per_device_train_batch_size=config['training']["per_device_train_batch_size"],
        remove_unused_columns=config['training']["remove_unused_columns"],
        fp16=config['training']["fp16"],
        dataloader_drop_last=config['training']["dataloader_drop_last"],
        group_by_length=config['training']["group_by_length"],
        
        gradient_checkpointing=config['training']["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config['training']["gradient_checkpointing_kwargs"],
        gradient_accumulation_steps=config['training']["gradient_accumulation_steps"],
        # torch_empty_cache_steps=config['training']["torch_empty_cache_steps"],
        dataloader_num_workers=config['training']["dataloader_num_workers"],

        per_device_eval_batch_size=config['training']["per_device_eval_batch_size"],
        evaluation_strategy=config['training']["evaluation_strategy"],
        eval_steps=config['training']["eval_steps"],
        
        predict_with_generate=config['training']["predict_with_generate"],
        generation_max_length=config['training']["generation_max_length"],
        report_to=config['training']["report_to"],
    )

    # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화 해줍니다.
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )

    # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수를 설정합니다.
    os.environ["WANDB_LOG_MODEL"]="end" # wandb에 가장 validation 점수가 좋은 checkpoint만 업로드하여 storage 절약.
    os.environ["WANDB_WATCH"]="false"
    # Hugging Face의 tokenizers 라이브러리가 병렬 처리(parallelism) 기능을 사용할지 여부
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    optimizer = torch.optim.AdamW(
        generate_model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config['training']['weight_decay'],
        amsgrad=False
    )

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks = [MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config['training']['warmup_steps'],
                num_training_steps=len(train_inputs_dataset) * config['training']['num_train_epochs']
            )
        )
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer

import pandas as pd

# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    # 클래스 초기화 메서드
    def __init__(self,
            bos_token: str, # 문장의 시작을 알리는 토큰
            eos_token: str, # 문장의 끝을 알리는 토큰
        ) -> None:

        self.bos_token = bos_token # 시작 토큰을 인스턴스 변수에 저장
        self.eos_token = eos_token # 종료 토큰을 인스턴스 변수에 저장

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    # 정적 메서드로, 클래스 인스턴스 없이 호출 가능
    def make_set_as_df(file_path, is_train = True):
        # is_train 플래그가 True이면 학습용 데이터로 처리
        if is_train:
            df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
            train_df = df[['fname','dialogue','summary']] # 'fname', 'dialogue', 'summary' 컬럼 선택
            return train_df # 생성된 학습 데이터프레임 반환
        # is_train 플래그가 False이면 테스트용 데이터로 처리
        else:
            df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
            test_df = df[['fname','dialogue']] # 'fname', 'dialogue' 컬럼 선택
            return test_df # 생성된 테스트 데이터프레임 반환

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset,is_test = False):
        # is_test 플래그가 True이면 테스트 데이터셋용 입력 생성
        if is_test:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = [self.bos_token] * len(dataset['dialogue']) # 디코더 입력은 시작 토큰(bos_token)으로만 구성 -> dialogue 개수만큼 bos_token 생성.
            return encoder_input.tolist(), list(decoder_input) # 인코더 입력과 디코더 입력을 리스트 형태로 반환
        # is_test 플래그가 False이면 학습/검증 데이터셋용 입력 생성
        else:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # 디코더 입력은 'summary' 앞에 시작 토큰(bos_token)을 추가하여 생성
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token) # 디코더 출력(레이블)은 'summary' 뒤에 종료 토큰(eos_token)을 추가하여 생성
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist() # 인코더 입력, 디코더 입력, 디코더 출력을 리스트 형태로 반환
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# import sys
# sys.path.append(
#     "/data/ephemeral/home/nlp-5/song/"
# )
# from src.dataset.dataset_base import *
# from src.dataset.preprocess import *
# from src.models.BART import *

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config, preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset


# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config, generate_model, tokenizer):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to(device),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path'] # submission 파일 경로
    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
    output.to_csv(os.path.join(result_path), index=False)

    return output
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig

# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer

def load_tokenizer_and_model_for_inference(config, device):
    tokenizer = AutoTokenizer.from_pretrained(
        config['inference']['ckt_dir']
    )
    model = BartForConditionalGeneration.from_pretrained(
        config['inference']['ckt_dir']
    ).to(device)
    return tokenizer, model

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

        # best 모델과 토크나이저 저장
        trainer.model.save_pretrained(config['inference']['ckt_dir'])
        tokenizer.save_pretrained(config['inference']['ckt_dir'])

        # inference 후 submission 파일 저장.
        _ = inference(config, trainer.model, tokenizer)
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

    args = parser.parse_args()

    # Load Configuration file
    config_path = os.path.join(
        project_dir,'src','configs',
        args.config # config 파일 이름을 설정
    )
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if not args.inference:
        main(loaded_config)
    else:
        device = "cuda:0" if torch.cuda.is_available else "cpu"
        tokenizer, generate_model = load_tokenizer_and_model_for_inference(loaded_config, device)
        inference(loaded_config, generate_model, tokenizer)

