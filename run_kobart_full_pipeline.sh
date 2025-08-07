#!/bin/bash
# KoBART 전체 파이프라인 실행 스크립트
# 모든 기능을 포함한 통합 실험 (데이터 증강, QLoRA, 특수 토큰, 후처리 등)

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 실험 시작 시간
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

echo -e "${CYAN}🚀 KoBART 전체 파이프라인 실험 시작${NC}"
echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
echo -e "⏰ 시작 시간: ${START_TIME_STR}"
echo -e "🎯 목표: KoBART로 모든 기능 테스트 및 성능 극대화"
echo -e "📋 포함 기능:"
echo -e "   ✅ 데이터 증강 (동의어 치환, 문장 재배열, 백트랜슬레이션)"
echo -e "   ✅ QLoRA 최적화 (4-bit 양자화)"
echo -e "   ✅ 특수 토큰 가중치 처리"
echo -e "   ✅ 전처리 정규화"
echo -e "   ✅ 후처리 파이프라인"
echo -e "   ✅ 최적화된 빔 서치"
echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"

# 로그 디렉토리 생성
LOG_DIR="logs/kobart_full_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# GPU 상태 확인
print_gpu_info() {
    echo -e "${BLUE}📊 GPU 상태:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || echo "GPU 정보를 가져올 수 없습니다."
    echo
}

# GPU 메모리 정리
cleanup_gpu() {
    echo -e "${YELLOW}🧹 GPU 메모리 정리 중...${NC}"
    /opt/conda/bin/python3 -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
" 2>/dev/null || true
    echo "✅ GPU 메모리 정리 완료"
    echo
}

# 초기 GPU 상태 확인
print_gpu_info

# KoBART 실험 설정 목록
declare -a experiments=(
    # 1. 베이스라인 (기본 KoBART)
    "01_baseline_kobart.yaml:1️⃣_KoBART_베이스라인:기본 설정으로 성능 기준점 확인"
    
    # 2. 데이터 증강 적용
    "01_simple_augmentation.yaml:2️⃣_데이터_증강:동의어 치환과 문장 재배열"
    
    # 3. 고성능 학습률
    "03_high_learning_rate.yaml:3️⃣_고성능_학습률:학습률 최적화"
    
    # 4. 배치 최적화
    "04_batch_optimization.yaml:4️⃣_배치_최적화:큰 배치로 안정적 학습"
    
    # 5. 전처리 정규화
    "04_text_normalization.yaml:5️⃣_텍스트_정규화:입력 텍스트 전처리"
    
    # 6. 특수 토큰 가중치
    "07_token_weighting.yaml:6️⃣_토큰_가중치:PII와 화자 토큰 보존"
    
    # 7. 빔 서치 최적화
    "08_beam_search_optimization/08a_diverse_beam_search.yaml:7️⃣_빔_서치_최적화:다양한 요약 생성"
    
    # 8. 백트랜슬레이션
    "09_backtranslation.yaml:8️⃣_백트랜슬레이션:한→영→한 데이터 증강"
    
    # 9. 모든 최적화 통합
    "10_combination_phase2/10c_all_optimizations.yaml:9️⃣_전체_통합:모든 기능 동시 적용"
)

# 실행 결과 추적
declare -a results=()
TOTAL_EXPERIMENTS=${#experiments[@]}
COMPLETED=0
FAILED=0

# 각 실험 실행
for i in "${!experiments[@]}"; do
    IFS=':' read -r config_file exp_name description <<<"${experiments[$i]}"
    
    EXPERIMENT_NUM=$((i + 1))
    echo -e "${PURPLE}🔬 실험 ${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}: ${exp_name}${NC}"
    echo -e "${WHITE}📄 설정: config/experiments/${config_file}${NC}"
    echo -e "${WHITE}📝 설명: ${description}${NC}"
    echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # 실험 시작 시간
    EXP_START_TIME=$(date +%s)
    EXP_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}🚀 실험 시작: ${EXP_START_TIME_STR}${NC}"
    
    # 파일명 정리
    exp_name_clean="${exp_name// /_}"
    exp_name_clean="${exp_name_clean//[^a-zA-Z0-9_]/}"
    LOG_FILE="${LOG_DIR}/exp_${EXPERIMENT_NUM}_${exp_name_clean}.log"
    
    # 설정 파일 존재 확인
    CONFIG_PATH="config/experiments/${config_file}"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${YELLOW}⚠️  설정 파일이 없습니다. KoBART 기본 설정으로 생성합니다.${NC}"
        
        # 디렉토리가 없으면 생성
        CONFIG_DIR=$(dirname "$CONFIG_PATH")
        mkdir -p "$CONFIG_DIR"
        
        # KoBART 기본 설정으로 복사
        cp config/experiments/01_baseline_kobart.yaml "$CONFIG_PATH" 2>/dev/null || \
        cp config/experiments/01_baseline.yaml "$CONFIG_PATH" 2>/dev/null || \
        echo -e "${RED}❌ 기본 설정 파일을 찾을 수 없습니다.${NC}"
    fi
    
    # 실험 실행
    if [ -f "$CONFIG_PATH" ]; then
        echo -e "${BLUE}🏃 실행 중... (로그: ${LOG_FILE})${NC}"
        
        # 실행 명령어 (KoBART 강제 적용)
        EXPERIMENT_CMD="/opt/conda/bin/python3 code/auto_experiment_runner.py \
            --config ${CONFIG_PATH} \
            --force-model digit82/kobart-summarization"
        
        if eval "$EXPERIMENT_CMD > ${LOG_FILE} 2>&1"; then
            EXP_END_TIME=$(date +%s)
            EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
            EXP_DURATION_MIN=$((EXP_DURATION / 60))
            EXP_DURATION_SEC=$((EXP_DURATION % 60))
            
            echo -e "${GREEN}✅ 실험 ${EXPERIMENT_NUM} 완료!${NC}"
            echo -e "⏱️  소요 시간: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초"
            
            # 결과 파일 확인
            if [ -f "./prediction/latest_output.csv" ]; then
                echo -e "📤 결과 파일: ./prediction/latest_output.csv"
            fi
            
            results+=("✅ ${exp_name}: ${EXP_DURATION_MIN}분 ${EXP_DURATION_SEC}초")
            COMPLETED=$((COMPLETED + 1))
        else
            echo -e "${RED}❌ 실험 ${EXPERIMENT_NUM} 실패!${NC}"
            echo -e "${YELLOW}📄 로그 확인: ${LOG_FILE}${NC}"
            
            # 에러 로그 일부 출력
            echo -e "${YELLOW}🔍 최근 에러:${NC}"
            tail -n 20 "$LOG_FILE" | grep -E "(ERROR|Error|Traceback|Exception)" | tail -n 5 || true
            
            results+=("❌ ${exp_name}: 실패")
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "${RED}❌ 설정 파일 생성 실패${NC}"
        results+=("❌ ${exp_name}: 설정 파일 없음")
        FAILED=$((FAILED + 1))
    fi
    
    echo
    
    # 다음 실험 전 GPU 정리 (마지막 실험 제외)
    if [ "$i" -lt $((TOTAL_EXPERIMENTS - 1)) ]; then
        echo -e "${YELLOW}⏸️  다음 실험 준비 중...${NC}"
        cleanup_gpu
        sleep 5
    fi
done

# 전체 소요 시간 계산
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

# 결과 요약
echo
echo -e "${CYAN}🎉 KoBART 전체 파이프라인 실험 완료!${NC}"
echo -e "${WHITE}════════════════════════════════════════════════════════${NC}"
echo -e "⏰ 종료 시간: ${END_TIME_STR}"
echo -e "⏱️  총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
echo -e "✅ 성공: ${COMPLETED}/${TOTAL_EXPERIMENTS}"
echo -e "❌ 실패: ${FAILED}/${TOTAL_EXPERIMENTS}"
echo
echo -e "${BLUE}📊 실험 결과 요약:${NC}"
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
for result in "${results[@]}"; do
    echo -e "  ${result}"
done

# 최종 결과 위치 안내
echo
echo -e "${GREEN}📁 결과 파일 위치:${NC}"
echo -e "  📤 최신 제출: ./prediction/latest_output.csv"
echo -e "  📋 실험 인덱스: ./prediction/experiment_index.csv"
echo -e "  📊 실험별 결과: ./prediction/*/output.csv"
echo -e "  📄 실험 로그: ${LOG_DIR}/"
echo -e "  🔬 상세 결과: outputs/auto_experiments/"
echo -e "  📈 WandB: https://wandb.ai/lyjune37-juneictlab/nlp-5"

# 실험 요약 파일 생성
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
{
    echo "KoBART 전체 파이프라인 실험 요약"
    echo "================================="
    echo "실행 시간: ${START_TIME_STR} ~ ${END_TIME_STR}"
    echo "총 소요 시간: ${TOTAL_HOURS}시간 ${TOTAL_MINUTES}분"
    echo "성공/실패: ${COMPLETED}/${FAILED}"
    echo
    echo "실험 결과:"
    for result in "${results[@]}"; do
        echo "  ${result}"
    done
    echo
    echo "테스트된 기능:"
    echo "- 데이터 증강 (동의어, 문장 재배열, 백트랜슬레이션)"
    echo "- QLoRA 최적화"
    echo "- 특수 토큰 가중치"
    echo "- 전처리 정규화"
    echo "- 후처리 파이프라인"
    echo "- 빔 서치 최적화"
} >"${SUMMARY_FILE}"

echo
echo -e "${WHITE}📝 실험 요약 저장: ${SUMMARY_FILE}${NC}"
echo
echo -e "${CYAN}💡 다음 단계:${NC}"
echo -e "  1. WandB에서 실험 결과 비교"
echo -e "  2. 최고 성능 실험의 설정 확인"
echo -e "  3. prediction/experiment_index.csv에서 ROUGE 점수 확인"
echo -e "  4. 최종 제출 파일 선택"
echo
echo -e "${GREEN}✨ KoBART 전체 기능 테스트 완료!${NC}"
