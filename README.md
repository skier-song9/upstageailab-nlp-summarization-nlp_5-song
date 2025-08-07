# Dialogue Summarization Competition

## Team

| ![송규헌](https://avatars.githubusercontent.com/u/113088511?v=4) | ![이상현](https://avatars.githubusercontent.com/u/48020255?v=4) | ![이영준](https://avatars.githubusercontent.com/u/97797643?v=4) | ![조은별](https://avatars.githubusercontent.com/u/178245805?v=4) | ![편아현](https://avatars.githubusercontent.com/u/83211745?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [송규헌](https://github.com/skier-song9)             |            [이상현](https://github.com/yourshlee)             |            [이영준](https://github.com/ProDevJune)             |            [조은별](https://github.com/eunbyul2)             |            [편아현](https://github.com/vusdkvus1)             |
|                            팀장                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |
|   프로젝트 리드, EDA, AI-hub 데이터 전처리, solar api 활용, baseline 코드 구현 및 발전, 논문 자료조사, huggingface space에 gradio app 배포      |     데이터 특성 분석 및 문제 정의, EDA 기여, 데이터 전처리/후처리 전략 실험 , 모델 실험    |   실험 자동화 환경 구축, 모델 실험    |   지시표현 EDA   |   baseline 코드 발전, optuna 하이퍼파라미터 실험, 모델 실험   |

## 0. Overview
### 🖥️ Environment
```bash
           .          .           .     .            .        root@instance-15691 
  .      .      *           .       .          .              ------------------- 
     .   . *      " Many a little makes a mickle...           OS: Ubuntu 20.04.6 LTS x86_64 
  .       .         .         Tikkeul moa Technologies "      Packages: 364 (dpkg) 
   *           .      _   .       *       .     . .           CPU: AMD Ryzen Threadripper 3960X (48) @ 3.800GHz
                     /#\                                      Memory: 7234MiB / 257656MiB (2%) 
                    /###\     /\                              GPU: NVIDIA GeForce RTX 3090 
                   /  ###\   /##\  /\                         CUDA version: 12.2 
                  /      #\ /####\/##\                        Python: 3.11.0 
                 /  /      /   # /  ##\                       torch: 2.6.0+cu124 
               // //  /\  /    _/  /  #\__                    torchvision: 0.21.0+cu124 
              // /   /  \     /   /    #\ \                   numpy: 1.26.4 
             /  \   / .. \   / /   _   { \ \                  pytorch-lightning: 2.5.2 
     /\     /    /\  ...  \_/   / / \   } \ |\                transformers: 4.54.0 
  _ /  \  /// / .\  ..%:.  /... /\ . \ {:  \ \                
 /.\ .\.\// \/... \.::::..... _/..\ ..\:|:. .\ \
/...\.../..:.\. ..:::::::..:..... . ...\{:... \ \                                     
 .:..\:..:::....:::;;;;;;::::::::.:::::.\}.....::%.:.                                 
::::...:::;;:::::;;;;;;;;;;;;;;:::::;;::{:::::::;;;:...
;;;;:::;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;];;;;;;;;;;:....
;;;;;;;;;;;;;;ii;;;;;;;;;;;;;;;;;;;;;;;;[;;;;;;;;;;;;;;....
```

### ⚙️ Requirements
- `UV`를 통해 간편하게 requirements.txt를 설치하세요.
```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# activate conda env
conda create -n nlp python==3.11 -y
# install dependencies
conda activate nlp
uv pip install -r requirements.txt
```

## 1. Competiton Info

### Overview

- [DialogSum](https://arxiv.org/abs/2105.06762) 논문의 데이터셋을 한국어로 번역한 데이터셋을 사용하는 대화 요약 경진대회이다. 


### Timeline

- July 25, 2025 - Start Date
- August 05, 2025 - Final submission deadline

## 2. Components

### ⚙️ Directory

e.g.
```
.
|-- README.md
|-- data
|-- huggingface_app
|   |-- app.py
|   |-- data
|   |-- requirements_app.txt
|   `-- solar_api.py
|-- outputs
|-- requirements.txt
|-- scripts
|   `-- multi_configs_exp.sh
`-- src
    |-- main_base.py
    |-- main_base_add_optuna.py
    |-- main_base_modified.py
    |-- configs
    |-- dataset
    |-- inference
    |-- models
    |-- notebooks
    |   `-- *.ipynb
    |-- trainer
    |-- trial
    `-- utils

```

## 3. Data descrption

### 📚 Dataset overview

- Dataset
    | data | size | columns | 
    |:----:|:----:|:-------:|
    | Train | 12457 | fname, dialogue, summary, topic |
    | Validation | 499 | fname, dialogue, summary, topic |
    | Test | 499 | fname, dialogue |
    - PII(Personal Identifier Information) masking : #Person1#, ..., #Person7#, #Address#, ..., #Email#.
- Summary 작성의 주요 기준
    1. 대화의 가장 중요한 정보를 전달
    2. 대화 길이의 20% 이내로 간략하게 요약
    3. 대화 내에 중요한 명명된 개체를 보존 (사람 이름, 기업명 등)
    4. 관찰자의 관점에서 작성 (화자의 의도를 이해하고 작성)
    5. 은어나 약어 없이 공식적으로 사용되는 언어로 작성

### 🧪 EDA

#### ⚙️ Tool

- Explore Data by gradio app : [HuggingFace Space](https://huggingface.co/spaces/song9/ai-lab-13-nlp-dialogsum)

#### 분석 내용

- Summary 작성 기준인 "대화 길이의 20% 이내로 간략하게 요약"에 따라 Tokenizer 별 Train/Validation/Test 데이터셋의 Dialogue, Summary 토큰 개수 분석 
    - test summary는 best submission의 summary를 사용
    - 사용한 토크나이저 : `upstage/solar-pro2-tokenizer`
    > 대체로 dialogue가 100~250 토큰으로 길이가 짧으며 summary의 길이도 짧다.

![length_eda](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main//images/length_eda.png)

- Summary 작성 기준인 "대화 길이의 20% 이내로 간략하게 요약"에 따라 Dialogue와 Summary 간 토큰 길이의 상관관계 분석
    - test summary는 best submission의 summary를 사용
    - 사용한 토크나이저 : `upstage/solar-pro2-tokenizer`
    > Train, Validation 모두 dialogue-summary 간 상관관계가 존재하는 것으로 판단할 수 있고, 따라서 test summary도 아래처럼 상관관계가 존재할 때 좋은 점수를 받을 것으로 예상할 수 있다.
    ```bass
    --- Train Dataset ---
    Pearson Correlation Coefficient: 0.6729
    P-value: 0.0000

    --- Validation Dataset ---
    Pearson Correlation Coefficient: 0.6624
    P-value: 0.0000

    --- Test Dataset ---
    Pearson Correlation Coefficient: 0.7884
    P-value: 0.0000
    ```   
    (빨간 점: compression_ratio 상위 99%, 하위 1%를 벗어나는 이상치)

![length_corr](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/images/length_corr.png)


### 🦾 Data Processing

- [src/dataset/preprocess.py](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/src/dataset/preprocess.py) 
    - `clean_text` 함수 : 줄바꿈 표현 정제, 'ㅎㅎ' 감정 표현 문장으로 대체, 중복 줄바꿈/공백 제거
    - `add_instruction` : config에 따라 topic 컬럼을 dialogue 앞에 추가할 수 있도록 설정.

- Data Augmentation
    - Back Translation : (Korean > English > Korean) 버전과 (Korean > English > Japanese > Korean) 버전 수행
    - AI-Hub 외부 데이터 활용 : [일상대화 한국어 멀티세션 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EC%9A%94%EC%95%BD&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=71678) Training, Validation 중 topic mediumcategory 비율을 맞춰 약 15000개 버전, 약 30000개 버전 준비.


## 4. Modeling

### Model descrition

- BART
    - `digit82/kobart-summarization` (base)
    - `cosmoquester/bart-ko-base`
    - `EbanLee/kobart-summary-v3`
- T5
    - `csebuetnlp/mT5_multilingual_XLSum`
    - `eenzeenee/t5-base-korean-summarization`
    - `paust/pko-flan-t5-large`


### Modeling Process

- refer to [src](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/tree/main/src)

## 5. Result

### Leader Board

![LB](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/images/LB.png)
- Public rank 3rd -> Final rank 5th (due to overfitting)

### Presentation

- [ppt.pdf](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/images/ppt.pdf)

## etc

### Meeting Log

- Notion을 활용하여 TODO-list와 Submission 제출 버전을 관리.
![todo](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/images/todo_notion.png)
![todo](https://github.com/AIBootcamp13/upstageailab-nlp-summarization-nlp_5/blob/main/images/submission_hub.png)


### Reference

- [Chen, Yulong, et al. "DialogSum: A real-life scenario dialogue summarization dataset." arXiv preprint arXiv:2105.06762 (2021).] (https://arxiv.org/abs/2105.06762)
- [cosmoquester/2021-dialogue-summary-competition](https://github.com/cosmoquester/2021-dialogue-summary-competition)
- AI-Hub [일상대화 한국어 멀티세션 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EC%9A%94%EC%95%BD&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=71678)
- [Upstage Solar API](https://console.upstage.ai/api/document-digitization/document-parsing)

