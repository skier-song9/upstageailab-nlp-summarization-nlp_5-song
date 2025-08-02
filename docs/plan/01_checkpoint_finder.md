# CheckpointFinder 설계 문서

## 📋 **목적**
현재 시스템에서 체크포인트 탐색 실패 문제를 해결하여 정확한 경로에서 학습된 모델을 찾아 추론에 사용할 수 있도록 한다.

## 🚨 **현재 문제점**

### **기존 코드 (auto_experiment_runner.py 353라인)**:
```python
# 잘못된 체크포인트 탐색
output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
checkpoint_dirs = list(output_dir.glob('checkpoint-*'))
```

### **문제 분석**:
- **잘못된 경로**: `outputs/` 에서 직접 체크포인트 탐색
- **실제 경로**: `outputs/dialogue_summarization_YYYYMMDD_HHMMSS/checkpoints/checkpoint-*`
- **결과**: "⚠️ 체크포인트를 찾을 수 없습니다." 에러 발생

## 🔧 **CheckpointFinder 설계**

### **주요 기능**
1. **정확한 경로 탐색**: 실제 체크포인트 저장 구조에 맞춤
2. **최신 실험 감지**: 여러 실험 중 가장 최근 것 선택
3. **베스트 체크포인트 선택**: 학습 단계별 체크포인트 중 최적 선택
4. **유효성 검증**: 체크포인트 파일 완전성 확인

### **클래스 구조**
```python
class CheckpointFinder:
    def __init__(self, base_output_dir: str = "outputs")
    def find_latest_checkpoint(self, experiment_id: str = None) -> Optional[Path]
    def _find_experiment_directories(self) -> List[Path]
    def _find_experiment_by_id(self, experiment_dirs: List[Path], experiment_id: str) -> Optional[Path]
    def _find_best_checkpoint(self, checkpoint_dirs: List[Path]) -> Path
    def validate_checkpoint(self, checkpoint_path: Path) -> bool
```

## 🗂️ **실제 파일 구조 분석**

### **현재 시스템의 체크포인트 저장 구조**:
```
📁 outputs/
├── 📁 dialogue_summarization_20250801_165443/    # 실험별 디렉토리
│   ├── 📁 checkpoints/                           # 체크포인트 저장소
│   │   ├── 📁 checkpoint-2000/                   # 학습 단계별 체크포인트
│   │   │   ├── 📄 pytorch_model.bin
│   │   │   ├── 📄 config.json
│   │   │   └── 📄 tokenizer.json
│   │   ├── 📁 checkpoint-2200/
│   │   ├── 📁 checkpoint-2400/
│   │   ├── 📁 checkpoint-2600/
│   │   └── 📁 checkpoint-2800/                   # 최신 체크포인트
│   ├── 📁 experiments/
│   ├── 📁 models/
│   └── 📁 results/
└── 📁 dialogue_summarization_20250801_132808/    # 다른 실험
    └── 📁 checkpoints/
        └── ...
```

## 🔍 **탐색 알고리즘**

### **1단계: 실험 디렉토리 탐색**
```python
def _find_experiment_directories(self) -> List[Path]:
    """outputs/dialogue_summarization_* 패턴으로 실험 디렉토리 찾기"""
    experiment_dirs = list(self.base_output_dir.glob("dialogue_summarization_*"))
    return [d for d in experiment_dirs if d.is_dir()]
```

### **2단계: 최신 실험 선택**
```python
def _find_experiment_by_id(self, experiment_dirs: List[Path], experiment_id: str) -> Optional[Path]:
    """실험 ID 매칭 또는 시간 기준 최신 선택"""
    # 특정 실험 ID가 없으면 가장 최근 실험 선택
    if experiment_dirs:
        return max(experiment_dirs, key=lambda p: p.stat().st_mtime)
    return None
```

### **3단계: 체크포인트 디렉토리 탐색**
```python
checkpoint_dir = target_dir / "checkpoints"
checkpoint_dirs = list(checkpoint_dir.glob("checkpoint-*"))
```

### **4단계: 최적 체크포인트 선택**
```python
def _find_best_checkpoint(self, checkpoint_dirs: List[Path]) -> Path:
    """checkpoint-숫자에서 가장 큰 숫자(최신) 선택"""
    numbered_checkpoints = []
    for cp_dir in checkpoint_dirs:
        try:
            number = int(cp_dir.name.split('-')[-1])  # checkpoint-2800 -> 2800
            numbered_checkpoints.append((number, cp_dir))
        except (ValueError, IndexError):
            continue
    
    if numbered_checkpoints:
        _, best_checkpoint = max(numbered_checkpoints, key=lambda x: x[0])
        return best_checkpoint
```

### **5단계: 유효성 검증**
```python
def validate_checkpoint(self, checkpoint_path: Path) -> bool:
    """필수 파일 존재 확인"""
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
    for file_name in required_files:
        if not (checkpoint_path / file_name).exists():
            return False
    return True
```

## 📊 **사용 예시**

### **auto_experiment_runner.py에서 사용**:
```python
# 기존 (실패)
output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
checkpoint_dirs = list(output_dir.glob('checkpoint-*'))  # 빈 리스트

# 수정 (성공)
from utils.checkpoint_finder import CheckpointFinder

checkpoint_finder = CheckpointFinder()
best_checkpoint = checkpoint_finder.find_latest_checkpoint(experiment_id)

if best_checkpoint and checkpoint_finder.validate_checkpoint(best_checkpoint):
    print(f"🎯 발견된 체크포인트: {best_checkpoint}")
    # 추론 진행
else:
    print("❌ 유효한 체크포인트를 찾을 수 없습니다.")
```

### **예상 로그 출력**:
```
🔍 체크포인트 탐색 시작: experiment_id=mt5_xlsum_ultimate_korean_qlora_08020154
📂 발견된 실험 디렉토리: 3개
  - outputs/dialogue_summarization_20250801_165443
  - outputs/dialogue_summarization_20250801_132808  
  - outputs/dialogue_summarization_20250801_140028
🎯 대상 실험 디렉토리: outputs/dialogue_summarization_20250801_165443
📁 체크포인트 디렉토리: outputs/dialogue_summarization_20250801_165443/checkpoints
🔢 발견된 체크포인트: checkpoint-2000, checkpoint-2200, checkpoint-2400, checkpoint-2600, checkpoint-2800
✅ 발견된 체크포인트: outputs/dialogue_summarization_20250801_165443/checkpoints/checkpoint-2800
✅ 체크포인트 유효성 검증 통과
```

## 🎯 **기대 효과**

1. **✅ 체크포인트 탐색 성공**: 100% 정확한 경로에서 찾기
2. **✅ 추론 단계 정상 실행**: 체크포인트를 찾아서 test.csv 추론 진행
3. **✅ 채점용 CSV 생성**: 추론 성공으로 제출 파일 자동 생성
4. **✅ 다중 실험 지원**: 여러 실험 중 원하는 것 선택 가능
5. **✅ 안정성 향상**: 유효성 검증으로 손상된 체크포인트 회피

## 🔗 **관련 파일**
- **구현**: `utils/checkpoint_finder.py`
- **사용**: `code/auto_experiment_runner.py`
- **테스트**: 실제 체크포인트 경로로 검증

---

**작성일**: 2025-08-02  
**상태**: 설계 완료, 구현 준비
