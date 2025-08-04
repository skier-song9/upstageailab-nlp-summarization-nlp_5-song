import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig
import warnings
tqdm.pandas()

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=True))

def eda_length(project_path, tokenizer, model=None):
    train_df = pd.read_csv(os.path.join(project_path, 'data','train.csv'))
    val_df = pd.read_csv(os.path.join(project_path, 'data','dev.csv'))
    test_df = pd.read_csv(os.path.join(project_path, 'data','test.csv'))
    sub_df = pd.read_csv(os.path.join(project_path, 'data','submission_best.csv')) # 48.02 점 submission 파일
    test_df['summary'] = sub_df['summary']

    # 토큰화 후 길이 분석
    print("Processing train_df...")
    train_df['dialogue_token_count'] = train_df['dialogue'].progress_apply(lambda x: count_tokens(x, tokenizer))
    train_df['summary_token_count'] = train_df['summary'].progress_apply(lambda x: count_tokens(x, tokenizer))

    print("\nProcessing val_df...")
    val_df['dialogue_token_count'] = val_df['dialogue'].progress_apply(lambda x: count_tokens(x, tokenizer))
    val_df['summary_token_count'] = val_df['summary'].progress_apply(lambda x: count_tokens(x, tokenizer))

    print("\nProcessing test_df...")
    test_df['dialogue_token_count'] = test_df['dialogue'].progress_apply(lambda x: count_tokens(x, tokenizer))
    test_df['summary_token_count'] = test_df['summary'].progress_apply(lambda x: count_tokens(x, tokenizer))

    train_df['token_compress_ratio'] = train_df['summary_token_count'] / train_df['dialogue_token_count']
    val_df['token_compress_ratio'] = val_df['summary_token_count'] / val_df['dialogue_token_count']
    test_df['token_compress_ratio'] = test_df['summary_token_count'] / test_df['dialogue_token_count']
    
    ### Dialogue의 최대 토큰 길이
    print(
        f"train dialogue 최대 토큰 길이: \t{max(train_df['dialogue_token_count'])}\n"
        f"validation dialogue 최대 토큰 길이: \t{max(val_df['dialogue_token_count'])}\n"
        f"test dialogue 최대 토큰 길이: \t{max(test_df['dialogue_token_count'])}\n"
    )
    max_encoder_length = max(max(train_df['dialogue_token_count']),max(val_df['dialogue_token_count']),max(test_df['dialogue_token_count']))

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8))
    fig.suptitle('Distribution of Token Counts and Compression Ratio')

    # Plot for dialogue_token_count
    sns.histplot(train_df['dialogue_token_count'], kde=True, ax=axes[0], color='skyblue', stat='density', label='train')
    sns.histplot(val_df['dialogue_token_count'], kde=True, ax=axes[0], color='lightcoral', stat='density', label='val')
    sns.histplot(test_df['dialogue_token_count'], kde=True, ax=axes[0], color='lightgreen', stat='density', label='test')
    axes[0].set_title('Train/Val/Test Dialogue Token Count')
    axes[0].set_xlabel('Token Count')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    sns.histplot(train_df['summary_token_count'], kde=True, ax=axes[1], color='skyblue', stat='density', label='train')
    sns.histplot(val_df['summary_token_count'], kde=True, ax=axes[1], color='lightcoral', stat='density', label='val')
    sns.histplot(test_df['summary_token_count'], kde=True, ax=axes[1], color='lightgreen', stat='density', label='test')
    axes[1].set_title('Train/Val/Test Summary Token Count')
    axes[1].set_xlabel('Token Count')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    sns.histplot(train_df['token_compress_ratio'], kde=True, ax=axes[2], color='skyblue', stat='density', label='train')
    sns.histplot(val_df['token_compress_ratio'], kde=True, ax=axes[2], color='lightcoral', stat='density', label='val')
    sns.histplot(test_df['token_compress_ratio'], kde=True, ax=axes[2], color='lightgreen', stat='density', label='test')
    axes[2].set_title('Train/Val/Test Compression Ratio')
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Density')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()

    return int(max_encoder_length)

def validate_model_enc_length(max_encoder_length: int, model_name: str) -> bool:
    """주어진 인코더 최대 길이가 모델이 수용 가능한지 검증하는 함수."""
    config = AutoConfig.from_pretrained(model_name)
    
    # 모델의 최대 시퀀스 길이를 가져옵니다.
    # 대부분의 모델은 max_position_embeddings를 사용합니다.
    model_max_length = int(getattr(config, 'max_position_embeddings', 0))
    
    if max_encoder_length > model_max_length:
        warnings.warn(
            f"⚠️  입력한 인코더 길이({max_encoder_length})가 모델의 최대 길이({model_max_length})를 초과합니다."
        )
        return False
    elif model_max_length == 0:
        warnings.warn(
            f"⚠️  {model_name} 모델의 인코더 최대 길이를 확인할 수 없습니다."
        )
        return False
    
    return True
