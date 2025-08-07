# CompetitionCSVManager 설계 문서

## 📋 **목적**
대회 채점용 CSV 파일을 baseline.py와 동일한 형식으로 생성하면서, 다중 실험 지원 및 실험 추적 기능을 제공한다.

## 🎯 **핵심 요구사항**

### **1. Baseline.py 호환성**
- **파일 형식**: `fname,summary` (동일)
- **저장 위치**: `./prediction/` 디렉토리 (동일)
- **파일명**: `output.csv` (동일)

### **2. 다중 실험 지원**  
- **실험별 구분**: 각 실험마다 독립된 폴더
- **타임스탬프**: 언제 생성되었는지 추적
- **덮어쓰기 방지**: 이전 실험 결과 보존

### **3. 편의 기능**
- **최신 결과**: `latest_output.csv` 빠른 접근
- **실험 추적**: `experiment_index.csv` 전체 관리
- **백업 기능**: `history/` 폴더에 히스토리 보관

## 🗂️ **생성될 파일 구조**

```
📁 prediction/
├── 📁 baseline_kobart_20250802_143022/         # 실험 1
│   ├── 📄 output.csv                           # 채점용 (표준 형식)
│   └── 📄 experiment_metadata.json             # 실험 정보
├── 📁 mt5_xlsum_20250802_151055/               # 실험 2  
│   ├── 📄 output.csv                           # 채점용 (표준 형식)
│   └── 📄 experiment_metadata.json
├── 📁 eenzeenee_t5_20250802_164233/            # 실험 3
│   ├── 📄 output.csv                           # 채점용 (표준 형식)
│   └── 📄 experiment_metadata.json
├── 📄 latest_output.csv                        # 최신 실험 결과 (편의용)
├── 📄 experiment_index.csv                     # 실험 인덱스 (추적용)
└── 📁 history/                                 # 백업 보관소
    ├── 📄 output_baseline_kobart_20250802_143022.csv
    ├── 📄 output_mt5_xlsum_20250802_151055.csv
    └── 📄 output_eenzeenee_t5_20250802_164233.csv
```

## 📊 **파일 내용 예시**

### **1. output.csv (채점용 - baseline.py와 동일)**
```csv
fname,summary
TREC002_00001,대화 참가자들이 여행 계획에 대해 논의하고 있다.
TREC002_00002,고객이 제품 구매 관련 문의를 하고 있다.
TREC002_00003,친구들이 저녁 식사 약속을 정하고 있다.
```

### **2. experiment_index.csv (실험 추적)**
```csv
experiment_name,folder_name,timestamp,submission_file,latest_file,created_at,rouge_combined,rouge1,rouge2,rougeL
mt5_xlsum,mt5_xlsum_20250802_151055,20250802_151055,./prediction/mt5_xlsum_20250802_151055/output.csv,./prediction/latest_output.csv,2025-08-02 15:10:55,0.579,0.254,0.095,0.230
baseline_kobart,baseline_kobart_20250802_143022,20250802_143022,./prediction/baseline_kobart_20250802_143022/output.csv,./prediction/latest_output.csv,2025-08-02 14:30:22,0.523,0.231,0.087,0.205
eenzeenee_t5,eenzeenee_t5_20250802_164233,20250802_164233,./prediction/eenzeenee_t5_20250802_164233/output.csv,./prediction/latest_output.csv,2025-08-02 16:42:33,0.467,0.198,0.078,0.189
```

### **3. experiment_metadata.json (상세 정보)**
```json
{
  "experiment_name": "mt5_xlsum",
  "timestamp": "20250802_151055",
  "created_at": "2025-08-02T15:10:55",
  "model_name": "csebuetnlp/mT5_multilingual_XLSum",
  "config_summary": {
    "learning_rate": 5e-05,
    "batch_size": 1,
    "num_epochs": 3
  },
  "metrics": {
    "eval_rouge1_f1": 0.254,
    "eval_rouge2_f1": 0.095,
    "eval_rougeL_f1": 0.230,
    "eval_rouge_combined_f1": 0.579
  },
  "submission_info": {
    "format": "fname,summary",
    "encoding": "utf-8"
  }
}
```

## 🔧 **클래스 설계**

### **주요 메서드**
```python
class CompetitionCSVManager:
    def __init__(self, prediction_base: str = "./prediction")
    
    # 핵심 기능
    def save_experiment_submission(self, experiment_name: str, result_df: pd.DataFrame, 
                                 config: Dict = None, metrics: Dict = None, timestamp: str = None) -> Dict[str, str]
    
    # 내부 기능
    def _save_experiment_metadata(self, metadata_path: Path, experiment_name: str, config: Dict, metrics: Dict, timestamp: str)
    def _save_to_history(self, submission_df: pd.DataFrame, experiment_name: str, timestamp: str) -> str
    def _update_experiment_index(self, experiment_name: str, experiment_folder: str, timestamp: str, metrics: Dict = None)
    def _print_generation_summary(self, result_paths: Dict[str, str], num_samples: int)
    
    # 조회 기능
    def get_latest_experiment(self) -> Optional[Dict]
    def list_all_experiments(self) -> pd.DataFrame
    def get_best_experiment_by_rouge(self) -> Optional[Dict]
```

## 🔄 **실행 흐름**

### **입력**
- `result_df`: DataFrame with columns ['fname', 'summary']
- `experiment_name`: "mt5_xlsum"
- `config`: 실험 설정 딕셔너리
- `metrics`: 성능 지표 딕셔너리

### **처리 과정**
1. **입력 검증**: fname, summary 컬럼 확인
2. **폴더 생성**: `prediction/{experiment_name}_{timestamp}/`
3. **채점용 CSV 저장**: `output.csv` (baseline.py 형식)
4. **메타데이터 저장**: `experiment_metadata.json`
5. **최신 파일 업데이트**: `latest_output.csv` (덮어쓰기)
6. **히스토리 백업**: `history/output_{experiment_name}_{timestamp}.csv`
7. **인덱스 업데이트**: `experiment_index.csv`에 실험 추가

### **출력**
```python
{
    'experiment_path': './prediction/mt5_xlsum_20250802_151055/output.csv',
    'latest_path': './prediction/latest_output.csv',
    'history_path': './prediction/history/output_mt5_xlsum_20250802_151055.csv',
    'metadata_path': './prediction/mt5_xlsum_20250802_151055/experiment_metadata.json',
    'experiment_folder': 'mt5_xlsum_20250802_151055'
}
```

## 📊 **사용 예시**

### **auto_experiment_runner.py에서 사용**
```python
from utils.competition_csv_manager import CompetitionCSVManager

csv_manager = CompetitionCSVManager()

# 추론 완료 후 호출
competition_paths = csv_manager.save_experiment_submission(
    experiment_name="mt5_xlsum",
    result_df=result_df,  # fname, summary 컬럼 포함
    config=config,
    metrics={"eval_rouge1_f1": 0.254, "eval_rouge2_f1": 0.095},
    timestamp=None  # 자동 생성
)

print(f"✅ 채점용 파일: {competition_paths['experiment_path']}")
print(f"✅ 최신 파일: {competition_paths['latest_path']}")
```

### **실험 조회**
```python
# 가장 최근 실험
latest = csv_manager.get_latest_experiment()
print(f"최근 실험: {latest['experiment_name']} → {latest['submission_file']}")

# 최고 성능 실험  
best = csv_manager.get_best_experiment_by_rouge()
print(f"최고 성능: {best['experiment_name']} (ROUGE: {best['rouge_combined']})")

# 전체 실험 목록
all_experiments = csv_manager.list_all_experiments()
print(all_experiments[['experiment_name', 'rouge_combined', 'created_at']])
```

## 🎯 **baseline.py와의 비교**

| 구분 | baseline.py | CompetitionCSVManager |
|------|------------|----------------------|
| **저장 위치** | `./prediction/output.csv` | ✅ `./prediction/{실험명}/output.csv` |
| **파일 형식** | `fname,summary` | ✅ 동일 |
| **다중 실험** | ❌ 덮어쓰기만 | ✅ 실험별 폴더 분리 |
| **최신 결과** | ❌ 없음 | ✅ `latest_output.csv` |
| **실험 추적** | ❌ 없음 | ✅ `experiment_index.csv` |
| **메타데이터** | ❌ 없음 | ✅ JSON 형태로 저장 |
| **백업** | ❌ 없음 | ✅ `history/` 폴더 |

## 📈 **기대 효과**

1. **✅ 대회 표준 준수**: baseline.py와 동일한 제출 형식
2. **✅ 다중 실험 지원**: 여러 실험 결과 동시 관리  
3. **✅ 추적 가능성**: 언제 어떤 실험으로 생성됐는지 기록
4. **✅ 편의성**: latest_output.csv로 최신 결과 빠른 접근
5. **✅ 안전성**: 이전 실험 결과 보존 및 백업
6. **✅ 분석 기능**: 실험 성능 비교 및 최적 모델 선택

---

**작성일**: 2025-08-02  
**상태**: 설계 완료, 구현 준비
