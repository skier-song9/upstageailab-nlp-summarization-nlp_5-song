import os
from torch.utils.data import Dataset

# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 수정: self.encoder_input은 이제 텐서가 아닌 BatchEncoding 객체(dict of lists)입니다.
        # 따라서 텐서 연산(.clone().detach())을 제거하고, 해당 인덱스의 리스트를 직접 반환합니다.
        item = {key: val[idx] for key, val in self.encoder_input.items()}
        
        # 수정: self.labels도 텐서가 아니므로, 텐서 연산을 제거합니다.
        item['labels'] = self.labels['input_ids'][idx]
        return item # 모델의 입력으로 사용될 딕셔너리 반환

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    # 클래스 초기화 메서드
    def __init__(self, encoder_input, labels, len):
        self.encoder_input = encoder_input # 토큰화된 인코더 입력을 인스턴스 변수에 저장
        self.labels = labels # 토큰화된 레이블(디코더 출력)을 인스턴스 변수에 저장
        self.len = len # 데이터셋의 길이를 인스턴스 변수에 저장

    # 특정 인덱스(idx)의 데이터를 가져오는 메서드
    def __getitem__(self, idx):
        # 수정: self.encoder_input은 이제 텐서가 아닌 BatchEncoding 객체(dict of lists)입니다.
        # 따라서 텐서 연산(.clone().detach())을 제거하고, 해당 인덱스의 리스트를 직접 반환합니다.
        item = {key: val[idx] for key, val in self.encoder_input.items()}

        # 수정: self.labels도 텐서가 아니므로, 텐서 연산을 제거합니다.
        item['labels'] = self.labels['input_ids'][idx]
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
        # 추론 데이터셋은 DataCollator를 사용하지 않으므로, 생성 시점에 텐서로 만들어도 무방합니다.
        # 따라서 기존 코드를 유지합니다.
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    # 데이터셋의 전체 길이를 반환하는 메서드
    def __len__(self):
        return self.len # __init__에서 저장한 길이 반환

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_dir, tokenizer):
    train_file_path = os.path.join(data_dir,'train.csv')
    val_file_path = os.path.join(data_dir,'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    encoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    # --- 토크나이저 수정 지점 --- #
    # 원인: return_tensors="pt"와 padding=False를 함께 사용하면, 길이가 다른 시퀀스로 텐서를 만들 수 없어 ValueError가 발생합니다.
    # 해결: return_tensors="pt"를 제거합니다. 이제 토크나이저는 텐서가 아닌 토큰 ID의 리스트를 반환합니다.
    #      패딩과 텐서 변환은 Trainer의 DataCollator가 배치 단위로 효율적으로 처리하게 됩니다.
    tokenized_encoder_inputs = tokenizer(
        encoder_input_train, 
        # return_tensors="pt", # 제거
        padding=False,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    tokenized_decoder_outputs = tokenizer(
        decoder_output_train, 
        # return_tensors="pt", # 제거
        padding=False,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_outputs, len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val, 
        # return_tensors="pt", # 제거
        padding=False,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False
    )
    val_tokenized_decoder_outputs = tokenizer(
        decoder_output_val, 
        # return_tensors="pt", # 제거
        padding=False,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False
    )

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_outputs, len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset