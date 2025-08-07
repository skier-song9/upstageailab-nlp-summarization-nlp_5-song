import pandas as pd
import os
import re
data_dir = '/data/ephemeral/home/nlp-5/song/data/'
daily_df_all_cleaned1 = pd.read_csv(os.path.join(data_dir, "ai_hub_02daily_sft.csv"))
daily_df_all_cleaned2 = pd.read_csv(os.path.join(data_dir, "ai_hub_02daily_dapt.csv"))
print(daily_df_all_cleaned1.columns)
print(daily_df_all_cleaned1.shape)
print(daily_df_all_cleaned2.columns)
print(daily_df_all_cleaned2.shape)

def jaccard_similarity_with_tokenizer(sentence1, sentence2, tokenizer):
    """자카드 유사도 측정
    J(A,B) = Intersection(A, B) / Union(A, B)
    > 토큰화 기준 자카드 유사도 측정 = 겹치는 토큰 개수 / 전체 토큰 개수

    :param _type_ sentence1: _description_
    :param _type_ sentence2: _description_
    :param _type_ tokenizer: _description_
    :return _type_: _description_
    """
    # 1. 문장을 토크나이저로 분리합니다.
    # AutoTokenizer 코드
    # tokens1 = set(tokenizer.tokenize(sentence1)) 
    # tokens2 = set(tokenizer.tokenize(sentence2))
    # Kiwi tokenizer 코드
    tokens1 = tokenizer.tokenize(sentence1)
    tokens1 = set([t.form for t in tokens1])
    tokens2 = tokenizer.tokenize(sentence2)
    tokens2 = set([t.form for t in tokens2])
    # 2. 파이썬의 set 자료형을 사용하여 교집합과 합집합을 구합니다.
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    # 3. 분모(합집합의 크기)가 0인 경우를 처리합니다.
    # 두 문장 모두 비어있을 경우 1.0을 반환합니다.
    if len(union) == 0:
        return 1.0
    # 4. 교집합의 크기를 합집합의 크기로 나누어 유사도를 계산합니다.
    return len(intersection) / len(union)

from multiprocessing import Pool, cpu_count
from kiwipiepy import Kiwi
import re
from tqdm import tqdm

def process_topic_list(tt):
    """
    하나의 토픽 문자열에 대해 유사도 기반으로 중복을 제거합니다.
    """
    # Kiwi 객체는 프로세스마다 생성해야 안전합니다.
    kiwi = Kiwi()
    
    topic_lists = tt.split(",")
    if not topic_lists or len(topic_lists) < 2:
        return ", ".join(topic_lists)

    i = 0
    while i < len(topic_lists):
        j = i + 1
        while j < len(topic_lists):
            sim = jaccard_similarity_with_tokenizer(topic_lists[i], topic_lists[j], kiwi)
            
            if sim >= 0.2:
                if len(topic_lists[i]) >= len(topic_lists[j]):
                    topic_lists.pop(j)
                else:
                    topic_lists.pop(i)
                    i -= 1
                    break
            else:
                j += 1
        i += 1
    
    return ", ".join(topic_lists)

def parallel_topic_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    멀티프로세싱을 사용하여 데이터프레임의 topic 컬럼을 처리합니다.
    """
    # 시스템의 CPU 코어 수를 가져와 병렬 처리를 위한 풀을 생성합니다.
    # num_cores = cpu_count()
    num_cores = 48
    with Pool(num_cores) as pool:
        # pool.map을 사용하여 각 토픽 문자열에 병렬로 함수를 적용합니다.
        # cleaned_topics = pool.map(process_topic_list, df['topic'].values)
        cleaned_topics = list(tqdm(pool.imap_unordered(process_topic_list, df['topic'].values), total=len(df)))
    
    # 처리된 결과를 데이터프레임에 할당합니다.
    df['topic'] = cleaned_topics
    return df

daily_df_topic_all_cleaned1 = parallel_topic_processing(daily_df_all_cleaned1.copy())
daily_df_topic_all_cleaned2 = parallel_topic_processing(daily_df_all_cleaned2.copy())

daily_df_topic_all_cleaned1 = daily_df_topic_all_cleaned1[['fname','dialogue','summary','topic']]
daily_df_topic_all_cleaned2 = daily_df_topic_all_cleaned2[['fname','dialogue','summary','topic']]

daily_df_topic_all_cleaned1.to_csv(os.path.join(data_dir, "ai_hub_02daily_sft_cleaned.csv"), index=False)
daily_df_topic_all_cleaned2.to_csv(os.path.join(data_dir, "ai_hub_02daily_dapt_cleaned.csv"), index=False)