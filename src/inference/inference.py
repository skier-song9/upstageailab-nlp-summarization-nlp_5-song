import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# import sys
# sys.path.append(
#     "/data/ephemeral/home/nlp-5/song/"
# )
from src.dataset.dataset_base import DatasetForInference
from src.dataset.preprocess import Preprocess
from src.models.AutoModels import *

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config, preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    # 수정: make_input은 이제 인코더 입력만 반환합니다.
    encoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False,
    )
    
    # 삭제: decoder_input은 더 이상 사용되지 않습니다.
    # test_tokenized_decoder_inputs = tokenizer(...)

    test_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_inputs_dataset


# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config, generate_model, tokenizer):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    generate_model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        for item in tqdm(dataloader, desc="Generating summaries"):
            text_ids.extend(item['ID'])
            
            # 개선: attention_mask를 모델 생성에 함께 사용합니다.
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                attention_mask=item['attention_mask'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams'],
            )
            
            # 개선: batch_decode와 skip_special_tokens=True를 사용하여 디코딩을 효율적으로 처리합니다.
            decoded_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            summary.extend(decoded_summaries)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    # skip_special_tokens=True로 대부분 처리되지만, 사용자 정의 토큰 제거 로직은 유지합니다.
    remove_tokens = config['inference'].get('remove_tokens', []) # .get()으로 키 존재 여부 처리
    preprocessed_summary = summary
    if remove_tokens:
        for token in remove_tokens:
            preprocessed_summary = [sentence.replace(token," ").strip() for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path'] # submission 파일 경로
    
    # 개선: os.path.dirname을 사용하여 디렉토리 경로를 얻고, exist_ok=True로 안전하게 생성합니다.
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        
    output.to_csv(result_path, index=False)
    print(f"Inference results saved to {result_path}")

    return output
