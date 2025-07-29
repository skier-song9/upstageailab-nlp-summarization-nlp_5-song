# `main_base.py` 워크플로우 문서

이 문서는 `main_base.py` 스크립트를 중심으로 한 전체 데이터 처리, 모델 학습 및 추론 파이프라인을 설명합니다.

---

## 1. 시작 및 설정 (main_base.py)

프로젝트의 진입점 역할을 하며, 전체 프로세스를 총괄합니다.

1.  **설정 파일 로드**:
    *   `argparse`를 통해 커맨드 라인에서 `--config` 인자로 YAML 설정 파일의 이름을 받습니다. (예: `config_base.yaml`)
    *   지정된 설정 파일을 로드하여 모든 하이퍼파라미터, 경로, 학습 옵션을 `config` 딕셔너리로 가져옵니다.

2.  **실행 모드 결정**:
    *   `--inference` 플래그의 존재 여부에 따라 학습 모드(`main(config)`) 또는 추론 모드를 실행합니다.

3.  **초기화**:
    *   `torch.device`를 설정하여 GPU 사용 가능 여부를 확인하고 장치를 할당합니다.
    *   `src.models.AutoModels.load_tokenizer_and_model_for_train` 함수를 호출하여 `config`에 명시된 사전 학습된 모델(예: `gogamza/kobart-base-v2`)과 해당 토크나이저를 로드합니다.

---

## 2. 데이터 준비 파이프라인 (학습 모드)

입력된 원시 데이터(CSV)를 모델이 학습할 수 있는 형태의 텐서로 변환하는 과정입니다.

### 2.1. 데이터 전처리 (src/dataset/preprocess.py)

*   `Preprocess` 클래스가 `config`에 정의된 `bos_token`과 `eos_token`으로 초기화됩니다.
*   `make_set_as_df()`: `train.csv`와 `dev.csv` 파일을 `pandas.DataFrame`으로 읽어옵니다.
*   `make_input()`: 데이터프레임에서 `dialogue`와 `summary` 컬럼을 추출하여 각각 파이썬 리스트로 반환합니다. 이 단계에서는 단순히 텍스트만 추출하며, 특별한 토큰을 수동으로 추가하지 않습니다.

### 2.2. 토큰화 및 데이터셋 생성 (src/dataset/dataset_base.py)

*   `prepare_train_dataset` 함수가 전체 데이터 준비 과정을 담당합니다.
*   **토큰화**:
    *   `tokenizer()`를 호출하여 `dialogue` (인코더 입력)와 `summary` (레이블) 텍스트 리스트를 토큰화합니다.
    *   이때 `padding=False` 옵션을 사용합니다. 패딩은 배치 단위로 `DataCollator`가 동적으로 처리하는 것이 훨씬 효율적이기 때문입니다.
    *   토큰화 결과는 `input_ids`와 `attention_mask`를 포함하는 딕셔너리 형태의 PyTorch 텐서입니다.
*   **PyTorch Dataset 래핑**:
    *   토큰화된 결과를 `DatasetForTrain` 또는 `DatasetForVal` 클래스로 감쌉니다.
    *   `__init__` 메서드는 토큰화된 인코더 입력(`input_ids`, `attention_mask`)과 레이블(`input_ids`)을 저장합니다.
    *   `__getitem__(idx)` 메서드는 특정 인덱스(`idx`)에 해당하는 샘플을 조회할 때 다음 형식의 딕셔너리를 반환합니다. 이는 `Trainer`가 요구하는 형식과 일치합니다.
        ```python
        {
          'input_ids': tensor([...]),      # 인코더 입력 ID
          'attention_mask': tensor([...]), # 인코더 어텐션 마스크
          'labels': tensor([...])          # 디코더 출력 (정답) ID
        }
        ```

---

## 3. 모델 학습 (src/trainer/trainer_base.py)

데이터 준비가 완료되면, `Seq2SeqTrainer`를 설정하고 모델 파인튜닝을 시작합니다.

1.  **`load_trainer_for_train` 함수 호출**:
    *   `main_base.py`에서 이 함수를 호출하여 `Trainer` 객체를 생성합니다.

2.  **`Seq2SeqTrainingArguments` 설정**:
    *   `config` 파일에 정의된 학습 관련 하이퍼파라미터(배치 크기, 학습률, 에포크 수, 로깅 및 저장 전략 등)를 사용하여 `Seq2SeqTrainingArguments` 객체를 생성합니다.

3.  **`DataCollatorForSeq2Seq` 정의**:
    *   `DataCollatorForSeq2Seq`는 `Trainer`의 핵심 구성 요소입니다.
    *   **역할**: `DataLoader`로부터 전달받은 개별 샘플 리스트(배치)를 모델이 처리할 수 있는 단일 텐서로 만듭니다.
    *   **동적 패딩**: 배치의 샘플들을 해당 배치에서 가장 긴 시퀀스의 길이에 맞춰 동적으로 패딩합니다. 이는 전체 데이터셋을 고정된 최대 길이로 패딩하는 것보다 메모리 및 계산 효율성이 높습니다.
    *   **`decoder_input_ids` 자동 생성**: `labels`를 오른쪽으로 한 칸 이동(right-shift)시켜 `decoder_input_ids`를 자동으로 생성합니다. 이 때문에 전처리 과정에서 수동으로 `bos_token`을 추가할 필요가 없습니다.

4.  **`Seq2SeqTrainer` 초기화**:
    *   준비된 모든 구성 요소(모델, 인자, 학습/검증 데이터셋, 토크나이저, 데이터 콜레이터)를 전달하여 `Seq2SeqTrainer` 객체를 생성합니다.

5.  **학습 시작**:
    *   `trainer.train()` 메서드를 호출하여 파인튜닝을 시작합니다.
    *   학습이 완료되면 `load_best_model_at_end=True` 설정에 따라 가장 성능이 좋았던 모델 체크포인트가 `trainer.model`에 로드됩니다.

6.  **모델 저장**:
    *   학습이 끝난 최적의 모델(`trainer.model`)과 토크나이저를 `config`에 지정된 경로(`inference.ckt_dir`)에 저장합니다.

---

## 4. 추론 (src/inference/inference.py)

학습된 모델을 사용하여 테스트 데이터에 대한 요약을 생성하고 결과를 파일로 저장합니다.

1.  **`inference` 함수 호출**:
    *   `main_base.py`의 학습 과정 마지막 단계에서 `inference()` 함수가 호출됩니다.

2.  **테스트 데이터 준비**:
    *   `prepare_test_dataset` 함수가 `test.csv` 파일을 로드하고, 학습 데이터와 동일한 방식으로 전처리 및 토큰화를 수행하여 `DatasetForInference` 객체를 생성합니다.

3.  **요약 생성**:
    *   모델을 평가 모드(`model.eval()`)로 전환합니다.
    *   테스트 데이터로더를 순회하며 배치 단위로 `model.generate()`를 호출합니다. 이때 `input_ids`와 `attention_mask`가 모델에 전달됩니다.
    *   생성된 요약문 ID 시퀀스를 `tokenizer.batch_decode(..., skip_special_tokens=True)`를 사용하여 텍스트로 변환합니다.

4.  **결과 저장**:
    *   생성된 요약문을 `pandas.DataFrame`으로 구성한 후, `config`에 지정된 경로(`inference.result_path`)에 `submission.csv` 파일로 저장합니다.
