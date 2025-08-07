# 구현 체크리스트

## 📋 **전체 구현 계획**

### **Phase 1: 핵심 컴포넌트 구현**
- [ ] `utils/checkpoint_finder.py` 구현
- [ ] `utils/competition_csv_manager.py` 구현
- [ ] 기본 동작 테스트

### **Phase 2: 시스템 통합**
- [ ] `code/auto_experiment_runner.py` 수정
- [ ] 통합 테스트 실행
- [ ] 체크포인트 탐색 기능 검증

### **Phase 3: 스크립트 개선**
- [ ] `run_main_5_experiments.sh` 수정
- [ ] 출력 형식 개선
- [ ] 에러 진단 기능 추가

### **Phase 4: 종합 테스트**
- [ ] 전체 시스템 통합 테스트
- [ ] 채점용 CSV 생성 검증
- [ ] 문서화 완료

## 🔧 **Phase 1: 핵심 컴포넌트 구현**

### **1.1 CheckpointFinder 구현**

#### **파일 생성**
```bash
touch utils/checkpoint_finder.py
```

#### **구현 체크리스트**
- [ ] 클래스 기본 구조 생성
- [ ] `find_latest_checkpoint()` 메서드 구현
- [ ] `_find_experiment_directories()` 메서드 구현
- [ ] `_find_experiment_by_id()` 메서드 구현  
- [ ] `_find_best_checkpoint()` 메서드 구현
- [ ] `validate_checkpoint()` 메서드 구현
- [ ] 로깅 기능 추가
- [ ] 에러 처리 강화

#### **테스트 방법**
```python
# test_checkpoint_finder.py
from utils.checkpoint_finder import CheckpointFinder

finder = CheckpointFinder()
checkpoint = finder.find_latest_checkpoint()
print(f"발견된 체크포인트: {checkpoint}")

if checkpoint:
    is_valid = finder.validate_checkpoint(checkpoint)
    print(f"유효성 검사: {is_valid}")
```

#### **검증 기준**
- [ ] 실제 체크포인트 경로에서 정상 탐색
- [ ] 여러 실험 디렉토리 중 최신 선택
- [ ] 체크포인트 번호별 정렬 (checkpoint-2800 > checkpoint-2600)
- [ ] 필수 파일 존재 여부 확인
- [ ] 에러 상황에서 graceful 처리

### **1.2 CompetitionCSVManager 구현**

#### **파일 생성**
```bash
touch utils/competition_csv_manager.py
```

#### **구현 체크리스트**
- [ ] 클래스 기본 구조 생성
- [ ] `save_experiment_submission()` 메서드 구현
- [ ] `_save_experiment_metadata()` 메서드 구현
- [ ] `_save_to_history()` 메서드 구현
- [ ] `_update_experiment_index()` 메서드 구현
- [ ] `_print_generation_summary()` 메서드 구현
- [ ] 조회 기능 구현 (`get_latest_experiment()`, `list_all_experiments()` 등)
- [ ] 폴더 자동 생성 기능
- [ ] 에러 처리 및 로깅

#### **테스트 방법**
```python
# test_csv_manager.py
import pandas as pd
from utils.competition_csv_manager import CompetitionCSVManager

# 테스트 데이터 생성
test_df = pd.DataFrame({
    'fname': ['test001', 'test002'],
    'summary': ['테스트 요약 1', '테스트 요약 2']
})

csv_manager = CompetitionCSVManager()
result = csv_manager.save_experiment_submission(
    experiment_name="test_experiment",
    result_df=test_df
)

print("생성된 파일들:", result)
```

#### **검증 기준**
- [ ] `./prediction/` 디렉토리 자동 생성
- [ ] 실험별 폴더 생성 (실험명_타임스탬프)
- [ ] `output.csv` 파일 올바른 형식 (fname,summary)
- [ ] `latest_output.csv` 덮어쓰기 동작
- [ ] `experiment_index.csv` 업데이트
- [ ] 히스토리 백업 생성
- [ ] 메타데이터 JSON 저장

## 🔧 **Phase 2: 시스템 통합**

### **2.1 AutoExperimentRunner 수정**

#### **수정 위치**
```bash
code/auto_experiment_runner.py
```

#### **수정 체크리스트**
- [ ] 새로운 imports 추가
- [ ] `__init__` 메서드에 컴포넌트 초기화 추가
- [ ] `run_single_experiment()` 메서드 추론 로직 수정
- [ ] `_run_test_inference()` 메서드 새로 추가
- [ ] `_try_post_training_inference()` 메서드 추가
- [ ] `_try_direct_inference()` 메서드 추가
- [ ] `_get_latest_metrics()` 메서드 추가
- [ ] 기존 체크포인트 탐색 코드 제거
- [ ] 에러 처리 강화

#### **수정 전후 코드 비교**
```python
# 수정 전 (실패하는 코드)
output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
checkpoint_dirs = list(output_dir.glob('checkpoint-*'))

# 수정 후 (성공하는 코드)  
best_checkpoint = self.checkpoint_finder.find_latest_checkpoint(experiment_id)
```

#### **검증 방법**
```bash
# 단일 실험 테스트
python code/auto_experiment_runner.py --config config/experiments/test_01_mt5_xlsum_1epoch.yaml

# 로그에서 확인해야 할 내용
# ✅ "체크포인트 탐색 시작"
# ✅ "발견된 체크포인트: ..."
# ✅ "Test 추론 실행"
# ✅ "채점용 CSV 파일 생성 완료"
```

### **2.2 통합 테스트**

#### **테스트 계획**
- [ ] **Step 1**: 1-epoch 짧은 실험으로 기본 동작 확인
- [ ] **Step 2**: 체크포인트 탐색 기능 단독 테스트
- [ ] **Step 3**: CSV 생성 기능 단독 테스트
- [ ] **Step 4**: 전체 파이프라인 통합 테스트

#### **Step 1: 기본 동작 테스트**
```bash
# 1 epoch 테스트 실험 실행
python code/auto_experiment_runner.py --config config/experiments/test_01_mt5_xlsum_1epoch.yaml --one-epoch

# 확인 사항
ls -la ./prediction/
cat ./prediction/experiment_index.csv
wc -l ./prediction/latest_output.csv
```

#### **Step 2: 체크포인트 테스트**
```python
# checkpoint_test.py
from utils.checkpoint_finder import CheckpointFinder

finder = CheckpointFinder()

# 실험 디렉토리 목록
exp_dirs = finder._find_experiment_directories()
print(f"실험 디렉토리: {exp_dirs}")

# 최신 체크포인트 찾기
checkpoint = finder.find_latest_checkpoint()
print(f"최신 체크포인트: {checkpoint}")

# 유효성 검사
if checkpoint:
    valid = finder.validate_checkpoint(checkpoint)
    print(f"유효성: {valid}")
```

#### **Step 3: CSV 생성 테스트**
```python
# csv_test.py  
import pandas as pd
from utils.competition_csv_manager import CompetitionCSVManager

# 가짜 추론 결과 생성
fake_results = pd.DataFrame({
    'fname': [f'TREC{i:03d}' for i in range(1, 11)],
    'summary': [f'테스트 요약 {i}' for i in range(1, 11)]
})

manager = CompetitionCSVManager()
paths = manager.save_experiment_submission(
    experiment_name="integration_test",
    result_df=fake_results
)

print("생성된 파일들:")
for key, path in paths.items():
    print(f"  {key}: {path}")
```

## 🔧 **Phase 3: 스크립트 개선**

### **3.1 run_main_5_experiments.sh 수정**

#### **수정 체크리스트**
- [ ] 실험별 채점용 파일 확인 로직 추가
- [ ] 최종 요약 섹션 대폭 개선
- [ ] 문제 진단 함수 추가
- [ ] 사용자 가이드 섹션 추가
- [ ] 실험 인덱스 기반 결과 표시
- [ ] ROUGE 점수 표시 및 권장 제출 파일

#### **테스트 방법**
```bash
# 전체 5개 실험 실행 (1 epoch 모드)
bash run_main_5_experiments.sh -1

# 확인 사항
# ✅ 각 실험 완료 후 채점용 파일 경로 표시
# ✅ 최종 요약에서 prediction/ 폴더 안내
# ✅ 제출 방법 3가지 안내
# ✅ 실험 인덱스 정보 표시
```

### **3.2 출력 형식 검증**

#### **기대되는 출력 예시**
```bash
✅ 실험 1 완료!
📁 생성된 채점용 파일들:
  📤 실험별 제출: ./prediction/baseline_kobart_20250802_143022/output.csv
  📤 최신 제출: ./prediction/latest_output.csv (251 줄, 14:30:22 생성)
  📋 실험 인덱스: ./prediction/experiment_index.csv
  ✅ 추론 및 채점용 파일 생성 성공
```

## 🔧 **Phase 4: 종합 테스트**

### **4.1 전체 시스템 통합 테스트**

#### **테스트 시나리오**
- [ ] **시나리오 1**: 정상적인 5개 실험 완료
- [ ] **시나리오 2**: 일부 실험 실패 상황
- [ ] **시나리오 3**: 체크포인트 없는 상황
- [ ] **시나리오 4**: 디스크 공간 부족 상황

#### **시나리오 1: 정상 완료**
```bash
bash run_main_5_experiments.sh

# 검증 체크리스트
# ✅ 5개 실험 모두 성공
# ✅ ./prediction/ 폴더에 5개 실험 폴더 생성
# ✅ latest_output.csv가 마지막 실험 결과
# ✅ experiment_index.csv에 5개 실험 기록
# ✅ history/ 폴더에 5개 백업 파일
```

#### **시나리오 2: 일부 실험 실패**
```bash
# 의도적으로 잘못된 설정으로 실험 실행
# config 파일을 잘못 수정하여 실패 유도

# 검증 체크리스트  
# ✅ 실패한 실험에 대한 진단 메시지 표시
# ✅ 성공한 실험들의 채점용 파일은 정상 생성
# ✅ 최종 요약에서 성공/실패 구분 표시
```

### **4.2 채점용 CSV 검증**

#### **파일 형식 검증**
```bash
# output.csv 형식 확인
head ./prediction/latest_output.csv
# 예상 결과:
# fname,summary
# TREC002_00001,대화 참가자들이 여행 계획에 대해 논의하고 있다.

# 파일 완전성 확인
wc -l ./prediction/latest_output.csv
# 예상: 251 (헤더 1줄 + 데이터 250줄)

# 인코딩 확인
file ./prediction/latest_output.csv
# 예상: UTF-8 text
```

#### **baseline.py와 동일성 검증**
```python
# compare_with_baseline.py
import pandas as pd

# 현재 시스템 결과
current_result = pd.read_csv('./prediction/latest_output.csv')

# baseline.py 형식 검증
assert list(current_result.columns) == ['fname', 'summary']
assert len(current_result) == 250  # test.csv 샘플 수
assert current_result['fname'].notna().all()
assert current_result['summary'].notna().all()

print("✅ baseline.py와 동일한 형식 검증 완료")
```

### **4.3 성능 검증**

#### **추론 속도 측정**
- [ ] 단일 실험당 추론 시간 측정
- [ ] GPU 메모리 사용량 모니터링
- [ ] 디스크 I/O 성능 확인

#### **품질 검증**
- [ ] 생성된 요약문 품질 육안 확인
- [ ] ROUGE 점수 합리적 범위 확인
- [ ] 빈 요약문이나 오류 결과 없는지 확인

## 📊 **최종 검증 체크리스트**

### **기능 검증**
- [ ] ✅ 체크포인트 탐색 100% 성공
- [ ] ✅ test.csv 추론 정상 실행
- [ ] ✅ 채점용 CSV 자동 생성
- [ ] ✅ 다중 실험 결과 구분
- [ ] ✅ 실험 추적 및 인덱싱
- [ ] ✅ 에러 상황 적절한 처리

### **파일 구조 검증**
- [ ] ✅ `./prediction/` 폴더 존재
- [ ] ✅ 실험별 폴더 생성 (실험명_타임스탬프)
- [ ] ✅ `output.csv` 올바른 형식
- [ ] ✅ `latest_output.csv` 최신 결과
- [ ] ✅ `experiment_index.csv` 실험 추적
- [ ] ✅ `history/` 폴더 백업

### **사용자 경험 검증**  
- [ ] ✅ 실시간 진행 상황 표시
- [ ] ✅ 최종 결과 명확한 요약
- [ ] ✅ 제출 방법 구체적 안내
- [ ] ✅ 에러 시 해결책 제시
- [ ] ✅ 성능 정보 및 추천

### **호환성 검증**
- [ ] ✅ baseline.py와 동일한 CSV 형식
- [ ] ✅ 기존 고도화 기능 유지
- [ ] ✅ WandB 연동 정상 동작
- [ ] ✅ 다중 실험 환경 지원

## 🎯 **성공 기준**

### **최소 성공 기준**
- 5개 실험 중 최소 3개 이상 성공
- 채점용 CSV 파일 정상 생성
- baseline.py와 동일한 파일 형식
- 에러 발생 시 명확한 원인 제시

### **완전 성공 기준**
- 5개 실험 모두 성공
- 모든 실험의 채점용 CSV 자동 생성
- 실험 추적 및 성능 비교 기능 동작
- 사용자 친화적인 결과 요약 제공

---

**작성일**: 2025-08-02  
**최종 업데이트**: 2025-08-02  
**상태**: 구현 준비 완료  
**예상 구현 시간**: 4-6시간
