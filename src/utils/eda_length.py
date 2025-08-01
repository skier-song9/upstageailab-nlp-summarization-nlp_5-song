import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
tqdm.pandas()

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=True))

def eda_length(project_path, tokenizer):
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

    train_df['len_compress_ratio'] = train_df['summary_len_count'] / train_df['dialogue_len_count']
    val_df['len_compress_ratio'] = val_df['summary_len_count'] / val_df['dialogue_len_count']
    test_df['len_compress_ratio'] = test_df['summary_len_count'] / test_df['dialogue_len_count']

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 9))
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

    sns.histplot(train_df['compress_ratio'], kde=True, ax=axes[2], color='skyblue', stat='density', label='train')
    sns.histplot(val_df['compress_ratio'], kde=True, ax=axes[2], color='lightcoral', stat='density', label='val')
    sns.histplot(test_df['compress_ratio'], kde=True, ax=axes[2], color='lightgreen', stat='density', label='test')
    axes[2].set_title('Train/Val/Test Compression Ratio')
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Density')
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()

    ### Dialogue의 최대 토큰 길이
    print(max(train_df['dialogue']))









