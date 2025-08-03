## 1. generate_config.ipynb

1. 학습시킬 모델 이름을 설정
2. 토크나이저 관련 설정 참고
	- (코드 수정 X) remove_tokens 를 추출한다. -> config 파일의 remove_tokens에 추가될 것.
3. project_dir 을 본인의 경로에 맞게 수정한다.
	```python
	# 모델의 구성 정보를 YAML 파일로 저장합니다.
	project_dir = "/data/ephemeral/home/nlp-5/auto1p/" # 여기에 올바른 경로 설정
	
	import os
	import sys
	sys.path.append(
		project_dir
	)
	from src.utils.utils import *
	```
4. 실험 컨셉에 따라 output_dir 이름 설정
5. config 파일에서 여러 번 작성해야 하는 파라미터를 미리 설정
	- `encoder_max_len`는 위에서 출력되는 시각화 및 print 결과를 보고 적절히 설정
		- 이 때, 토큰 길이 EDA 중 `Token indices sequence length is longer than the specified maximum sequence length for this model (555 > 512). Running this sequence through the model will result in indexing errors` 와 같은 경고가 발생했다면 해당 모델의 토크나이저는 최대 512 까지만 처리가 가능한 것이다. 이런 경우, 우리 경진대회용 데이터셋을 제대로 입력하지 못해 해당 모델은 사용하기 어렵다.
	```python
	save_eval_log_steps = 100 # save_step은 eval, log step의 배수여야 한다. 같이 맞춰주는 것이 편하다.
	train_batch_size = 8
	inference_batch_size = 8 # eval batch size 도 동일하게 설정됨
	encoder_max_len = 1500
	decoder_max_len = 170 # inference max length 도 동일하게 설정됨
	```
6. 나머지 config 설정들은 주석으로 설명을 달았으니 참고하시면 될 것 같습니다.


## 2. 코드 내에서 경로 설정

#### - src/main_base.py
- line 9 : `project_dir` 경로를 본인에 맞게 수정


## 3. 코드 개발 시 세팅
- 전처리, 모델링 등 코드를 개발한 후, 오류를 확인하고 싶은 경우
  `python src/main_base.py --config config.yaml --practice True` 로 파일을 실행하면 됩니다.
	- practice 모드는 train_df에서 256개, val_df와 test_df에서는 10개만 사용하므로 train, evaluation 과정이 짧아 1epoch가 더 빠르게 수행되는 코드 확인용 모드입니다.
	- ⚠️ 이때, train_batch_size는 256보다 작아야 하고, evaluation_batch_size 및 inference_batch_size는 10보다 작아야 합니다.

## 4. requirements.txt 업데이트
- 가상환경 실행 후 , `uv pip install -r requirements.txt` 실행하여 추가된 의존성 설치.

## 5. 주요 해결 사항
- T5 계열 모델과 BART 계열 모델의 decoder 처리 방식이 달라 오류가 발생했었다.
	> - **T5 계열 모델**: T5는 모델 내부적으로 `labels`를 받아서 `decoder_input_ids`를 자동으로 생성합니다. 즉, `labels`를 오른쪽으로 한 칸 시프트(shift)하고 `<pad>` 토큰을 추가하여 디코더의 입력으로 사용합니다. 따라서 `Trainer`에 `input_ids`와 `labels`만 전달해도 정상적으로 학습이 진행됩니다.
    
	- **BART 계열 모델**: BART는 T5와 달리, `labels`와 별개로 `decoder_input_ids`를 명시적으로 전달받아야 합니다. `Trainer`가 `labels`만 전달할 경우, BART 모델은 `decoder_input_ids`가 없기 때문에 손실 함수(`loss function`) 계산 시 차원 불일치(dimension mismatch) 오류가 발생합니다.

	> 해결 : BART 계열 모델을 처리할 때 tokenizer가 bos token과 eos token을 추가하지 않도록 기존 tokenizer 클래스를 오버라이딩하여 해결했다.