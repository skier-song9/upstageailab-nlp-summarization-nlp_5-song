import pandas as pd

# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    # 클래스 초기화 메서드
    def __init__(self,
            bos_token: str, # 문장의 시작을 알리는 토큰
            eos_token: str, # 문장의 끝을 알리는 토큰
        ) -> None:

        self.bos_token = bos_token # 시작 토큰을 인스턴스 변수에 저장
        self.eos_token = eos_token # 종료 토큰을 인스턴스 변수에 저장

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    # 정적 메서드로, 클래스 인스턴스 없이 호출 가능
    def make_set_as_df(file_path, is_train = True):
        # is_train 플래그가 True이면 학습용 데이터로 처리
        if is_train:
            df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
            # train_df = df[['fname','dialogue','summary']] # 이전 코드: 'fname', 'dialogue', 'summary' 컬럼 선택
            train_df = df[['id','dialogue','summary']] # 'id', 'dialogue', 'summary' 컬럼을 선택하여 학습 데이터프레임 생성
            return train_df # 생성된 학습 데이터프레임 반환
        # is_train 플래그가 False이면 테스트용 데이터로 처리
        else:
            df = pd.read_csv(file_path) # CSV 파일을 읽어 데이터프레임 생성
            # test_df = df[['fname','dialogue']] # 이전 코드: 'fname', 'dialogue' 컬럼 선택
            test_df = df[['id','dialogue']] # 'id', 'dialogue' 컬럼을 선택하여 테스트 데이터프레임 생성
            return test_df # 생성된 테스트 데이터프레임 반환

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset,is_test = False):
        # is_test 플래그가 True이면 테스트 데이터셋용 입력 생성
        if is_test:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = [self.bos_token] * len(dataset['dialogue']) # 디코더 입력은 시작 토큰(bos_token)으로만 구성 -> dialogue 개수만큼 bos_token 생성.
            return encoder_input.tolist(), list(decoder_input) # 인코더 입력과 디코더 입력을 리스트 형태로 반환
        # is_test 플래그가 False이면 학습/검증 데이터셋용 입력 생성
        else:
            encoder_input = dataset['dialogue'] # 인코더 입력으로 'dialogue' 컬럼 사용
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # 디코더 입력은 'summary' 앞에 시작 토큰(bos_token)을 추가하여 생성
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token) # 디코더 출력(레이블)은 'summary' 뒤에 종료 토큰(eos_token)을 추가하여 생성
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist() # 인코더 입력, 디코더 입력, 디코더 출력을 리스트 형태로 반환

