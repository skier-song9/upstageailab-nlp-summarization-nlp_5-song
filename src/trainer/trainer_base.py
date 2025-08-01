from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
import torch
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
import wandb
import os
import numpy as np
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
import re

def remove_origin_special_tokens(decoded_preds, decoded_labels, remove_tokens):
    replaced_predictions = list(decoded_preds)
    replaced_labels = list(decoded_labels)
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]
    replaced_predictions = [re.sub(r'\s+', ' ', sentence) for sentence in replaced_predictions]
    replaced_labels = [re.sub(r'\s+', ' ', sentence)  for sentence in replaced_labels]  
    return replaced_predictions, replaced_labels      

def compute_metrics(pred, config, tokenizer:AutoTokenizer, eval_tokenizer:AutoTokenizer=None):
    preds, labels = pred
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 생성된 summary를 decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # id가 -100인 input_ids는 padding 토큰으로 변환
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    # preds.argmax(-1) : 모델의 예측 결과에서 가장 확률이 높은 토큰 ID를 선택
    # skip_special_tokens=False : 현재 task에서는 special token이 summary에 포함되어야 하기 때문에 False로 설정. True면 디코딩 과정에서 special token을 제거함.
    decoded_preds = tokenizer.batch_decode(preds.argmax(-1), skip_special_tokens=False)

    # 앞에서 skip_special_tokens=False 했기 때문에 따로 저장해둔 remove_tokens를 제거해야 한다.
    # 수동으로 정의된 제거 토큰들을 디코딩된 문자열에서 제거합니다.
    replaced_predictions, replaced_labels = remove_origin_special_tokens(decoded_preds, decoded_labels, config['inference']['remove_tokens'])

    # eval_tokenizer가 있을 경우, 해당 토크나이저로 텍스트를 재처리하여 ROUGE를 계산합니다.
    if eval_tokenizer is None:
        # 디코딩된 문자열을 다시 토큰화하고 공백으로 재결합.
        #    이는 ROUGE 계산 시 토큰 경계를 명확히 하기 위함.
        retokenized_preds = [" ".join(tokenizer.tokenize(sentence)) for sentence in replaced_predictions]
        retokenized_labels = [" ".join(tokenizer.tokenize(sentence)) for sentence in replaced_labels]
    else: # eval_tokenizer로 다시 토큰화하고 공백으로 재결합
        retokenized_preds = [" ".join(eval_tokenizer.tokenize(sentence)) for sentence in replaced_predictions]
        retokenized_labels = [" ".join(eval_tokenizer.tokenize(sentence)) for sentence in replaced_labels]

    # 토큰 기준 Rouge 점수 계산.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) # 한국어 특성 상 stemmer 사용 안 함.
    scores = [scorer.score(label, pred) for label, pred in zip(retokenized_labels, retokenized_preds)]
    
    # 평균 f-점수를 계산하고 반환합니다.
    avg_scores = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        avg_scores[key] = np.mean([s[key].fmeasure for s in scores])
    
    result = {key.replace('rouge', 'rouge-'): value for key, value in avg_scores.items()}
    result['rouge-mean'] = np.mean(list(result.values()))
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
        bf16=config['training']["bf16"],
        dataloader_drop_last=config['training']["dataloader_drop_last"],
        group_by_length=config['training']["group_by_length"],
        
        gradient_checkpointing=config['training']["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config['training']["gradient_checkpointing_kwargs"],
        gradient_accumulation_steps=config['training']["gradient_accumulation_steps"],
        torch_empty_cache_steps=config['training']["torch_empty_cache_steps"],
        dataloader_num_workers=config['training']["dataloader_num_workers"],

        per_device_eval_batch_size=config['training']["per_device_eval_batch_size"],
        eval_strategy=config['training']["evaluation_strategy"],
        eval_steps=config['training']["eval_steps"],
        
        predict_with_generate=config['training']["predict_with_generate"],
        generation_max_length=config['training']["generation_max_length"],
        report_to=config['training']["report_to"],
    )

    if config['training']["report_to"] in ['all', 'wandb']:
        # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화 해줍니다.
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=config['wandb']['name'],
        )

    # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수를 설정합니다.
    os.environ["WANDB_LOG_MODEL"]="false" # wandb에 가장 validation 점수가 좋은 checkpoint만 업로드하여 storage 절약.
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

    ### evaluation 용 tokenizer가 설정되어 있다면 해당 토크나이저로 validation 점수 계산
    eval_tokenizer = None
    if config['general'].get("eval_tokenizer", False) and len(config['general']['eval_tokenizer']) > 1:
        eval_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['general']['eval_tokenizer'])
        eval_tokenizer.remove_tokens = list(eval_tokenizer.special_tokens_map.values())
        special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
        eval_tokenizer.add_special_tokens(special_tokens_dict)

    # T5, BART 등 서로 다른 계열의 모델에서 decoder input을 만들기 위한 DataCollator
    '''
    - 경고1
    UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:254.)
        - 배치의 시퀀스 길이를 맞추기 위해 패딩(padding)을 해야 하는데, 어떤 토큰 ID를 사용해야 할지 명확하게 지정되지 않았을 때 발생
    
    '''
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=generate_model,
        padding=True,
    )

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        data_collator=data_collator, # DataCollator
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(pred, config, tokenizer, eval_tokenizer),
        callbacks = [MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config['training']['warmup_steps'],
                num_training_steps=len(train_inputs_dataset) * config['training']['num_train_epochs'],
                num_cycles=config['training'].get("num_cycles", 1)
            )
        )
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer