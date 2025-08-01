import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import re

# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config, generate_model, tokenizer, test_df, summ_test_dataset, val_flag=False):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    def inference_collate_fn(batch):
        # batch의 labels가 None으로 넘어오면 input_ids와 attention_mask만 텐서 타입을 보장하여 묶는다.
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    # generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)
    dataloader = DataLoader(summ_test_dataset, batch_size=config['inference']['batch_size'], shuffle=False, collate_fn=inference_collate_fn)

    all_summary = []
    generate_model.eval()
    generate_model.gradient_checkpointing_disable()
    for item in tqdm(dataloader):
        with torch.no_grad():
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device), # Encoder Input
                decoder_start_token_id=tokenizer.pad_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id, # Decoder Begin Token ID
                early_stopping=config['inference']['early_stopping'], # True로 설정하면 모든 빔이 eos_token에 도달했을 때 생성을 조기 종료
            )
            decoded_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            # skip_special_tokens=False : 현재 task에서는 special token이 summary에 포함되어야 하기 때문에 False로 설정. True면 디코딩 과정에서 special token을 제거함.
            all_summary.extend(decoded_ids)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    preprocessed_summary = all_summary
    for remove_token in config['inference']['remove_tokens']:
        preprocessed_summary = [sentence.replace(remove_token, " ") for sentence in preprocessed_summary]
    preprocessed_summary = [re.sub(r'\s+', ' ', sentence) for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_df['fname'],
            "summary" : preprocessed_summary,
        }
    )

    if val_flag:
        val_file_path = os.path.join(config['general']['data_path'], config['general']['val_data'])
        # merge with val_df
        val_df = pd.read_csv(val_file_path)
        output = output.merge(val_df[['fname','dialogue','topic']], on='fname')

    result_path = config['inference']['result_path'] # submission 파일 경로

    if val_flag:
        result_path = os.path.dirname(result_path)
        result_path = os.path.join(result_path, "val_inference.csv")

    if not os.path.exists(result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
    output.to_csv(os.path.join(result_path), index=False)

    return output