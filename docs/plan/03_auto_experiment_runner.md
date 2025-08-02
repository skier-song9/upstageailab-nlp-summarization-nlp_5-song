# AutoExperimentRunner 수정 사항

## 📋 **목적**
현재 `auto_experiment_runner.py`의 추론 실패 문제를 해결하고, 새로운 CheckpointFinder와 CompetitionCSVManager를 통합하여 완전한 test.csv 추론 및 채점용 CSV 생성을 구현한다.

## 🚨 **현재 문제점**

### **1. 체크포인트 탐색 실패**
```python
# 현재 코드 (353라인)
output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
checkpoint_dirs = list(output_dir.glob('checkpoint-*'))  # 빈 리스트 반환

if checkpoint_dirs:  # False
    # 추론 실행 (실행되지 않음)
else:
    print("⚠️ 체크포인트를 찾을 수 없습니다.")  # 여기서 종료
```

### **2. 추론 단계 건너뛰기**
- 체크포인트를 못 찾아서 test.csv 추론이 실행되지 않음
- 결과적으로 채점용 CSV 파일이 생성되지 않음

### **3. 에러 처리 부족**
- 추론 실패 시 적절한 폴백 메커니즘 없음
- 사용자에게 명확한 실패 원인 제공 안됨

## 🔧 **수정 방안**

### **1. 새로운 컴포넌트 통합**

#### **초기화 부분 수정**
```python
class AutoExperimentRunner:
    def __init__(self, base_config_path: str = "config/base_config.yaml", 
                 output_dir: str = "outputs/auto_experiments"):
        # 기존 초기화...
        
        # 🆕 새로 추가: 체크포인트 탐색기와 CSV 관리자
        from utils.checkpoint_finder import CheckpointFinder
        from utils.competition_csv_manager import CompetitionCSVManager
        
        self.checkpoint_finder = CheckpointFinder()
        self.csv_manager = CompetitionCSVManager()
        
        print(f"🔧 체크포인트 탐색기 및 CSV 관리자 초기화 완료")
```

### **2. 추론 로직 강화**

#### **기존 추론 로직 (실패)**
```python
# 학습 완료 후
if process.returncode == 0:
    print(f"\n📊 Test 추론 시작: {experiment_id}")
    
    # 🚨 문제: 잘못된 경로에서 체크포인트 탐색
    output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
    checkpoint_dirs = list(output_dir.glob('checkpoint-*'))  # 실패
    
    if checkpoint_dirs:  # False
        # 추론 코드 (실행되지 않음)
    else:
        print("⚠️ 체크포인트를 찾을 수 없습니다.")
```

#### **수정된 추론 로직 (성공)**
```python
# 학습 완료 후
if process.returncode == 0:
    print(f"\n🎉 학습 완료! Test 추론 시작: {experiment_id}")
    
    try:
        # 🔧 수정: 정확한 체크포인트 탐색
        best_checkpoint = self.checkpoint_finder.find_latest_checkpoint(experiment_id)
        
        if best_checkpoint and self.checkpoint_finder.validate_checkpoint(best_checkpoint):
            print(f"🎯 발견된 체크포인트: {best_checkpoint}")
            
            # 🆕 강화된 추론 실행 (2단계 폴백)
            submission_info = self._run_test_inference(
                experiment_id=experiment_id,
                checkpoint_path=best_checkpoint,
                config=config
            )
            
            result = self._collect_results(config, Path(config_path).stem)
            result.update(submission_info)
            
        else:
            print("❌ 유효한 체크포인트를 찾을 수 없습니다.")
            result = self._collect_results(config, Path(config_path).stem)
            result['inference_error'] = "No valid checkpoint found"
            
    except Exception as inf_e:
        print(f"❌ 추론 실행 중 예외: {inf_e}")
        result = self._collect_results(config, Path(config_path).stem)
        result['inference_error'] = str(inf_e)
```

### **3. 2단계 폴백 추론 시스템**

```python
def _run_test_inference(self, experiment_id: str, checkpoint_path: Path, config: Dict) -> Dict[str, Any]:
    """Test 데이터 추론 및 채점용 CSV 생성"""
    print(f"📊 Test 추론 실행: {experiment_id}")
    
    # 방법 1: post_training_inference 사용 시도
    submission_info = self._try_post_training_inference(experiment_id, checkpoint_path, config)
    if submission_info:
        return submission_info
    
    # 방법 2: run_inference.py 직접 호출
    return self._try_direct_inference(experiment_id, checkpoint_path, config)
```

#### **방법 1: post_training_inference 사용**
```python
def _try_post_training_inference(self, experiment_id: str, checkpoint_path: Path, config: Dict) -> Optional[Dict]:
    """post_training_inference.py 사용"""
    try:
        from post_training_inference import generate_submission_after_training
        
        # 추론 실행
        submission_path = generate_submission_after_training(
            experiment_name=experiment_id,
            model_path=str(checkpoint_path),
            config_dict=config
        )
        
        # 결과 DataFrame 로드
        result_df = pd.read_csv(submission_path)
        
        # 🆕 채점용 CSV 생성 (다중 실험 지원)
        competition_paths = self.csv_manager.save_experiment_submission(
            experiment_name=experiment_id,
            result_df=result_df,
            config=config,
            metrics=self._get_latest_metrics(config)
        )
        
        return {
            'submission_path': submission_path,
            'competition_paths': competition_paths,
            'inference_method': 'post_training_inference'
        }
        
    except Exception as e:
        print(f"❌ post_training_inference 실행 실패: {e}")
        return None
```

#### **방법 2: run_inference.py 직접 호출**
```python
def _try_direct_inference(self, experiment_id: str, checkpoint_path: Path, config: Dict) -> Dict[str, Any]:
    """run_inference.py 직접 호출"""
    import subprocess
    import sys
    
    print(f"🔄 대안 추론 방법 사용: run_inference.py")
    
    temp_output = f"outputs/temp_inference_{experiment_id}.csv"
    
    inference_cmd = [
        sys.executable,
        str(path_manager.resolve_path("code/run_inference.py")),
        "--model_path", str(checkpoint_path),
        "--input_file", "data/test.csv",
        "--output_file", temp_output,
        "--batch_size", "16"
    ]
    
    try:
        inference_process = subprocess.run(
            inference_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        
        if inference_process.returncode == 0:
            print("✅ 대안 추론 성공")
            
            # 결과 파일 로드 및 채점용 CSV 생성
            if Path(temp_output).exists():
                result_df = pd.read_csv(temp_output)
                
                competition_paths = self.csv_manager.save_experiment_submission(
                    experiment_name=experiment_id,
                    result_df=result_df,
                    config=config,
                    metrics=self._get_latest_metrics(config)
                )
                
                # 임시 파일 정리
                Path(temp_output).unlink()
                
                return {
                    'submission_path': temp_output,
                    'competition_paths': competition_paths,
                    'inference_method': 'run_inference_direct'
                }
        else:
            return {'inference_error': inference_process.stderr}
            
    except subprocess.TimeoutExpired:
        return {'inference_error': 'Inference timeout (1 hour)'}
    except Exception as e:
        return {'inference_error': f'Direct inference failed: {str(e)}'}
```

## 📊 **수정 전후 비교**

| 구분 | 수정 전 | 수정 후 |
|------|---------|---------|
| **체크포인트 탐색** | ❌ 잘못된 경로 | ✅ CheckpointFinder 정확한 탐색 |
| **추론 실행** | ❌ 체크포인트 못 찾아서 건너뛰기 | ✅ 2단계 폴백 시스템 |
| **채점용 CSV** | ❌ 생성되지 않음 | ✅ CompetitionCSVManager 자동 생성 |
| **에러 처리** | ❌ 단순 에러 메시지 | ✅ 상세한 실패 원인 및 대안 제시 |
| **결과 추적** | ❌ 추론 성공 여부만 | ✅ 생성된 파일 경로 및 메타데이터 |

## 🎯 **예상 실행 결과**

### **성공 케이스**
```
🎉 학습 완료! Test 추론 시작: mt5_xlsum_ultimate_korean_qlora_08020154

🔍 체크포인트 탐색 시작: experiment_id=mt5_xlsum_ultimate_korean_qlora_08020154
📂 발견된 실험 디렉토리: 1개
  - outputs/dialogue_summarization_20250801_165443
🎯 대상 실험 디렉토리: outputs/dialogue_summarization_20250801_165443
📁 체크포인트 디렉토리: outputs/dialogue_summarization_20250801_165443/checkpoints
🔢 발견된 체크포인트: checkpoint-2000, checkpoint-2200, checkpoint-2400, checkpoint-2600, checkpoint-2800
✅ 발견된 체크포인트: outputs/dialogue_summarization_20250801_165443/checkpoints/checkpoint-2800
✅ 체크포인트 유효성 검증 통과

📊 Test 추론 실행: mt5_xlsum_ultimate_korean_qlora_08020154
✅ post_training_inference 성공

✅ 채점용 CSV 파일 생성 완료!
📊 처리된 샘플 수: 250
📁 생성된 파일들:
  📤 실험별 제출: ./prediction/mt5_xlsum_ultimate_korean_qlora_08020154_20250802_151055/output.csv
  📤 최신 제출: ./prediction/latest_output.csv
  📋 실험 인덱스: ./prediction/experiment_index.csv
  💾 히스토리 백업: ./prediction/history/output_mt5_xlsum_ultimate_korean_qlora_08020154_20250802_151055.csv
  📄 메타데이터: ./prediction/mt5_xlsum_ultimate_korean_qlora_08020154_20250802_151055/experiment_metadata.json
```

### **폴백 케이스**
```
⚠️ post_training_inference import 실패: ModuleNotFoundError
🔄 대안 추론 방법 사용: run_inference.py
실행 명령: python code/run_inference.py --model_path outputs/dialogue_summarization_20250801_165443/checkpoints/checkpoint-2800 --input_file data/test.csv --output_file outputs/temp_inference_mt5_xlsum.csv --batch_size 16
✅ 대안 추론 성공

✅ 채점용 CSV 파일 생성 완료!
(동일한 결과 출력)
```

## 🔗 **관련 파일**

### **수정 대상**
- `code/auto_experiment_runner.py` (주요 수정)

### **새로 추가**
- `utils/checkpoint_finder.py`
- `utils/competition_csv_manager.py`

### **기존 활용**
- `code/post_training_inference.py`
- `code/run_inference.py`
- `core/inference.py`

## 📈 **기대 효과**

1. **✅ 추론 성공률 100%**: 정확한 체크포인트 탐색으로 추론 실패 해결
2. **✅ 채점용 CSV 자동 생성**: 모든 실험에서 대회 표준 형식 파일 생성
3. **✅ 안정성 향상**: 2단계 폴백으로 추론 실패 위험 최소화
4. **✅ 사용자 경험 개선**: 명확한 진행 상황 및 결과 파일 경로 제공
5. **✅ 다중 실험 지원**: 여러 실험 결과를 체계적으로 관리

---

**작성일**: 2025-08-02  
**상태**: 설계 완료, 구현 준비
