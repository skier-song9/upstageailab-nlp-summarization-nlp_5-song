# 실행 스크립트 개선 사항

## 📋 **목적**
`run_main_5_experiments.sh` 스크립트를 개선하여 채점용 CSV 파일 생성 상황을 실시간으로 확인하고, 최종 결과를 명확하게 요약 제공한다.

## 🎯 **개선 목표**

1. **실시간 채점용 파일 확인**: 각 실험 완료 후 생성된 파일들 확인
2. **최종 결과 요약**: 전체 실험 완료 후 채점용 파일 위치 안내
3. **사용자 가이드**: 어떤 파일을 대회에 제출해야 하는지 안내
4. **문제 진단**: 채점용 파일 생성 실패 시 원인 파악

## 🔧 **수정 사항**

### **1. 각 실험 완료 후 파일 확인 추가**

#### **기존 코드**
```bash
if eval "$EXPERIMENT_CMD > ${LOG_FILE} 2>&1"; then
    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
    
    echo -e "${GREEN}✅ 실험 ${EXPERIMENT_NUM} 완료!${NC}"
    echo -e "⏱️  소요 시간: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초"
    
    # 기존에는 여기서 끝
fi
```

#### **수정된 코드**
```bash
if eval "$EXPERIMENT_CMD > ${LOG_FILE} 2>&1"; then
    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
    EXP_DURATION_MIN=$((EXP_DURATION / 60))
    EXP_DURATION_SEC=$((EXP_DURATION % 60))

    echo -e "${GREEN}✅ 실험 ${EXPERIMENT_NUM} 완료!${NC}"
    echo -e "⏱️  소요 시간: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초"
    
    # 🆕 채점용 파일 생성 확인
    echo -e "${BLUE}📁 생성된 채점용 파일들:${NC}"
    
    # 현재 시간 기준으로 최근 생성된 폴더 찾기
    if ls ./prediction/*_$(date +%Y%m%d)* 2>/dev/null | tail -1 >/dev/null; then
        latest_exp_folder=$(ls -td ./prediction/*_$(date +%Y%m%d)* 2>/dev/null | head -1)
        if [ -n "$latest_exp_folder" ] && [ -f "$latest_exp_folder/output.csv" ]; then
            echo -e "  📤 실험별 제출: ${latest_exp_folder}/output.csv"
        else
            echo -e "  ⚠️  실험별 제출 파일을 찾을 수 없습니다"
        fi
    else
        echo -e "  ⚠️  오늘 날짜의 실험 폴더를 찾을 수 없습니다"
    fi
    
    # 최신 파일 확인
    if [ -f "./prediction/latest_output.csv" ]; then
        echo -e "  📤 최신 제출: ./prediction/latest_output.csv"
        # 파일 크기도 표시
        file_size=$(wc -l < "./prediction/latest_output.csv")
        echo -e "      (${file_size} 줄, $(date -r ./prediction/latest_output.csv '+%H:%M:%S') 생성)"
    else
        echo -e "  ❌ 최신 제출 파일이 생성되지 않았습니다"
    fi
    
    # 실험 인덱스 확인
    if [ -f "./prediction/experiment_index.csv" ]; then
        echo -e "  📋 실험 인덱스: ./prediction/experiment_index.csv"
        # 최근 실험 3개 표시
        if [ $(wc -l < ./prediction/experiment_index.csv) -gt 1 ]; then
            echo -e "  🔍 최근 실험:"
            tail -n +2 ./prediction/experiment_index.csv | head -3 | while IFS=',' read -r exp_name folder_name timestamp file_path rest; do
                echo -e "    - $exp_name → $file_path"
            done
        fi
    else
        echo -e "  ❌ 실험 인덱스가 생성되지 않았습니다"
    fi
    
    # 간단한 추론 성공 여부 확인
    if [ -f "./prediction/latest_output.csv" ] && [ $(wc -l < "./prediction/latest_output.csv") -gt 1 ]; then
        echo -e "  ✅ 추론 및 채점용 파일 생성 성공"
    else
        echo -e "  ❌ 추론 또는 채점용 파일 생성 실패"
        # 로그에서 에러 확인
        if grep -q "체크포인트를 찾을 수 없습니다" "$LOG_FILE"; then
            echo -e "      원인: 체크포인트 탐색 실패"
        elif grep -q "추론 실행 중 예외" "$LOG_FILE"; then
            echo -e "      원인: 추론 실행 에러"
        else
            echo -e "      원인: 로그 파일 확인 필요 - $LOG_FILE"
        fi
    fi
fi
```

### **2. 최종 결과 요약 대폭 개선**

#### **기존 최종 요약**
```bash
echo -e "${GREEN}📁 결과 파일 위치:${NC}"
echo -e "  - 실험 로그: ${LOG_DIR}/"
echo -e "  - 실험 요약: outputs/auto_experiments/experiment_summary.json"
echo -e "  - 개별 결과: outputs/auto_experiments/experiments/"
echo -e "  - 모델 체크포인트: outputs/auto_experiments/"
echo -e "  - WandB 프로젝트: https://wandb.ai/lyjune37-juneictlab/nlp-5"
```

#### **수정된 최종 요약**
```bash
echo
echo -e "${CYAN}🎉 5개 실험 모두 완료!${NC}"
echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
echo -e "⏰ 종료 시간: ${END_TIME_STR}"
echo -e "⏱️  총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
echo
echo -e "${BLUE}📊 실험 결과 요약:${NC}"
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for result in "${results[@]}"; do
    echo -e "  ${result}"
done
echo
echo -e "${GREEN}📁 최종 결과 파일 위치:${NC}"
echo -e "  📤 채점용 파일들: ./prediction/"
echo -e "  📋 실험 추적: ./prediction/experiment_index.csv"
echo -e "  📊 최신 제출: ./prediction/latest_output.csv"
echo -e "  💾 백업 히스토리: ./prediction/history/"
echo -e "  📄 실험 로그: ${LOG_DIR}/"
echo -e "  🔬 상세 결과: outputs/auto_experiments/"
echo -e "  📈 WandB: https://wandb.ai/lyjune37-juneictlab/nlp-5"

# 🆕 최종 채점용 파일 요약
echo
echo -e "${CYAN}🏆 채점용 파일 최종 요약:${NC}"
echo -e "${WHITE}──────────────────────────────────────${NC}"

# 실험 인덱스 기반 요약
if [ -f "./prediction/experiment_index.csv" ]; then
    total_experiments=$(tail -n +2 ./prediction/experiment_index.csv | wc -l)
    echo -e "📊 총 실험 수: $total_experiments"
    
    # 최고 성능 실험 (ROUGE 점수 기준, 간단 버전)
    echo -e "🥇 실험 목록 (최신순):"
    tail -n +2 ./prediction/experiment_index.csv | head -5 | while IFS=',' read -r exp_name folder_name timestamp file_path latest_file created_at rouge_combined rest; do
        echo -e "   📋 $exp_name"
        echo -e "      📁 $file_path"
        if [ -n "$rouge_combined" ] && [ "$rouge_combined" != "0" ]; then
            echo -e "      📈 ROUGE: $rouge_combined"
        fi
        echo -e "      🕐 $created_at"
        echo
    done
    
    # 성능이 가장 좋은 실험 찾기 (간단 버전)
    if [ $(tail -n +2 ./prediction/experiment_index.csv | wc -l) -gt 0 ]; then
        best_experiment=$(tail -n +2 ./prediction/experiment_index.csv | head -1)
        best_exp_name=$(echo "$best_experiment" | cut -d',' -f1)
        best_file_path=$(echo "$best_experiment" | cut -d',' -f4)
        echo -e "🏆 권장 제출 파일:"
        echo -e "   $best_exp_name → $best_file_path"
    fi
else
    echo -e "⚠️  실험 인덱스 파일을 찾을 수 없습니다."
    echo -e "   📁 ./prediction/ 폴더를 직접 확인하세요."
fi

echo
echo -e "${WHITE}📝 채점 제출 방법:${NC}"
echo -e "  ${YELLOW}1. 최신 결과 사용:${NC}"
echo -e "     cp ./prediction/latest_output.csv submission.csv"
echo -e "  ${YELLOW}2. 특정 실험 결과 사용:${NC}"
echo -e "     cp ./prediction/{실험명}_{타임스탬프}/output.csv submission.csv"
echo -e "  ${YELLOW}3. 실험 비교 후 선택:${NC}"
echo -e "     cat ./prediction/experiment_index.csv"
echo -e "     # ROUGE 점수를 확인하여 최고 성능 실험 선택"
echo
echo -e "${GREEN}✨ 모든 실험 완료! 위 경로에서 제출할 파일을 선택하세요.${NC}"
```

### **3. 문제 진단 함수 추가**

```bash
# 채점용 파일 생성 문제 진단 함수
diagnose_submission_issues() {
    local log_file="$1"
    local experiment_name="$2"
    
    echo -e "${YELLOW}🔍 채점용 파일 생성 문제 진단:${NC}"
    
    # 체크포인트 관련 문제
    if grep -q "체크포인트를 찾을 수 없습니다" "$log_file"; then
        echo -e "  ❌ 체크포인트 탐색 실패"
        echo -e "     💡 해결 방법: outputs/dialogue_summarization_*/checkpoints/ 확인"
    fi
    
    # 추론 관련 문제
    if grep -q "추론 실행 중 예외" "$log_file"; then
        echo -e "  ❌ 추론 실행 에러"
        echo -e "     💡 해결 방법: $log_file 에서 상세 에러 확인"
    fi
    
    # CSV 생성 관련 문제  
    if grep -q "CSV 파일 생성" "$log_file"; then
        echo -e "  ❌ CSV 생성 에러"
        echo -e "     💡 해결 방법: prediction/ 폴더 권한 확인"
    fi
    
    # 메모리 관련 문제
    if grep -q "CUDA out of memory\|OutOfMemoryError" "$log_file"; then
        echo -e "  ❌ GPU 메모리 부족"
        echo -e "     💡 해결 방법: 배치 크기 줄이기 또는 GPU 정리"
    fi
    
    # 일반적인 해결책
    echo -e "  💡 일반적인 해결책:"
    echo -e "     1. 로그 파일 확인: cat $log_file | tail -50"
    echo -e "     2. GPU 상태 확인: nvidia-smi"
    echo -e "     3. 디스크 공간 확인: df -h"
    echo -e "     4. 권한 확인: ls -la ./prediction/"
}

# 실험 실패 시 진단 호출
handle_experiment_error() {
    local exp_name="$1"
    local log_file="$2"
    local exp_num="$3"
    
    echo -e "${RED}❌ 실험 $exp_num 실패: $exp_name${NC}"
    echo -e "${YELLOW}📄 로그 파일: $log_file${NC}"
    
    # 🆕 문제 진단 추가
    diagnose_submission_issues "$log_file" "$exp_name"
    
    # 기존 에러 처리 로직...
}
```

## 📊 **수정 전후 비교**

| 구분 | 수정 전 | 수정 후 |
|------|---------|---------|
| **실험별 결과** | ✅/❌ 상태만 표시 | ✅ 생성된 파일 경로와 크기 상세 표시 |
| **문제 진단** | 로그 파일 위치만 제공 | ✅ 구체적인 실패 원인과 해결책 제시 |
| **최종 요약** | 시스템 내부 파일 위치 | ✅ 사용자 관점의 채점용 파일 안내 |
| **제출 가이드** | 없음 | ✅ 구체적인 제출 방법 3가지 제시 |
| **성능 정보** | 없음 | ✅ ROUGE 점수 및 권장 제출 파일 |

## 🎯 **예상 실행 결과**

### **각 실험 완료 후**
```bash
✅ 실험 2 완료!
⏱️  소요 시간: 45분 23초
📁 생성된 채점용 파일들:
  📤 실험별 제출: ./prediction/mt5_xlsum_20250802_151055/output.csv
  📤 최신 제출: ./prediction/latest_output.csv
      (251 줄, 15:10:55 생성)
  📋 실험 인덱스: ./prediction/experiment_index.csv
  🔍 최근 실험:
    - baseline_kobart → ./prediction/baseline_kobart_20250802_143022/output.csv
    - mt5_xlsum → ./prediction/mt5_xlsum_20250802_151055/output.csv
  ✅ 추론 및 채점용 파일 생성 성공
```

### **최종 완료 후**
```bash
🎉 5개 실험 모두 완료!
════════════════════════════════════════════════════════
⏰ 종료 시간: 2025-08-02 18:45:30
⏱️  총 소요 시간: 3시간 42분

📊 실험 결과 요약:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ baseline_kobart: 38분 15초
  ✅ mt5_xlsum: 45분 23초  
  ✅ eenzeenee_t5: 32분 47초
  ✅ high_lr: 29분 18초
  ✅ batch_opt: 34분 12초

📁 최종 결과 파일 위치:
  📤 채점용 파일들: ./prediction/
  📋 실험 추적: ./prediction/experiment_index.csv
  📊 최신 제출: ./prediction/latest_output.csv
  💾 백업 히스토리: ./prediction/history/
  📄 실험 로그: logs/main_experiments_20250802_150803/
  🔬 상세 결과: outputs/auto_experiments/
  📈 WandB: https://wandb.ai/lyjune37-juneictlab/nlp-5

🏆 채점용 파일 최종 요약:
──────────────────────────────────────
📊 총 실험 수: 5
🥇 실험 목록 (최신순):
   📋 batch_opt
      📁 ./prediction/batch_opt_20250802_184530/output.csv
      📈 ROUGE: 0.623
      🕐 2025-08-02 18:45:30

   📋 high_lr
      📁 ./prediction/high_lr_20250802_172140/output.csv
      📈 ROUGE: 0.598
      🕐 2025-08-02 17:21:40

   📋 eenzeenee_t5
      📁 ./prediction/eenzeenee_t5_20250802_164233/output.csv
      📈 ROUGE: 0.587
      🕐 2025-08-02 16:42:33

🏆 권장 제출 파일:
   batch_opt → ./prediction/batch_opt_20250802_184530/output.csv

📝 채점 제출 방법:
  1. 최신 결과 사용:
     cp ./prediction/latest_output.csv submission.csv
  2. 특정 실험 결과 사용:
     cp ./prediction/{실험명}_{타임스탬프}/output.csv submission.csv
  3. 실험 비교 후 선택:
     cat ./prediction/experiment_index.csv
     # ROUGE 점수를 확인하여 최고 성능 실험 선택

✨ 모든 실험 완료! 위 경로에서 제출할 파일을 선택하세요.
```

### **실험 실패 시**
```bash
❌ 실험 3 실패: eenzeenee_t5
📄 로그 파일: logs/main_experiments_20250802_150803/experiment_3_eenzeenee_t5.log
🔍 채점용 파일 생성 문제 진단:
  ❌ 체크포인트 탐색 실패
     💡 해결 방법: outputs/dialogue_summarization_*/checkpoints/ 확인
  💡 일반적인 해결책:
     1. 로그 파일 확인: cat logs/main_experiments_20250802_150803/experiment_3_eenzeenee_t5.log | tail -50
     2. GPU 상태 확인: nvidia-smi
     3. 디스크 공간 확인: df -h
     4. 권한 확인: ls -la ./prediction/
```

## 📈 **기대 효과**

1. **✅ 실시간 모니터링**: 각 실험별 채점용 파일 생성 상황 즉시 확인
2. **✅ 문제 조기 발견**: 실험 실패 시 즉시 원인 파악 및 해결책 제시
3. **✅ 사용자 친화적**: 대회 참가자 관점에서 필요한 정보만 명확하게 제공
4. **✅ 제출 편의성**: 구체적인 파일 경로와 제출 방법 안내
5. **✅ 성능 비교**: ROUGE 점수 기반 최적 모델 추천

---

**작성일**: 2025-08-02  
**상태**: 설계 완료, 구현 준비
