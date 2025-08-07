#!/usr/bin/env python3
"""
통합 테스트 스크립트

새로 구현한 CheckpointFinder와 CompetitionCSVManager를 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_checkpoint_finder():
    """CheckpointFinder 테스트"""
    print("\n=== CheckpointFinder 테스트 ===")
    try:
        from utils.checkpoint_finder import CheckpointFinder
        
        finder = CheckpointFinder()
        checkpoint = finder.find_latest_checkpoint()
        
        if checkpoint:
            print(f"✅ 체크포인트 발견: {checkpoint}")
            print(f"   경로 존재: {checkpoint.exists()}")
            print(f"   유효성: {finder.validate_checkpoint(checkpoint)}")
        else:
            print("❌ 체크포인트를 찾을 수 없습니다.")
            
        return checkpoint is not None
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def test_competition_csv_manager():
    """CompetitionCSVManager 테스트"""
    print("\n=== CompetitionCSVManager 테스트 ===")
    try:
        from utils.competition_csv_manager import CompetitionCSVManager
        import pandas as pd
        
        # 테스트 데이터
        test_df = pd.DataFrame({
            'fname': ['TEST_001', 'TEST_002', 'TEST_003'],
            'summary': ['요약 1', '요약 2', '요약 3']
        })
        
        manager = CompetitionCSVManager()
        
        # 실험 결과 저장
        result_paths = manager.save_experiment_submission(
            experiment_name="integration_test",
            result_df=test_df
        )
        
        print("✅ 생성된 파일들:")
        for key, path in result_paths.items():
            if Path(path).exists():
                print(f"   {key}: {path} ✓")
            else:
                print(f"   {key}: {path} ✗")
                
        # 최신 실험 조회
        latest = manager.get_latest_experiment()
        if latest:
            print(f"\n✅ 최근 실험: {latest['experiment_name']}")
            
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """통합 테스트"""
    print("\n=== 통합 테스트 ===")
    
    # auto_experiment_runner가 제대로 수정되었는지 확인
    try:
        with open(project_root / "code" / "auto_experiment_runner.py", 'r') as f:
            content = f.read()
            
        checks = {
            "CheckpointFinder import": "from utils.checkpoint_finder import CheckpointFinder" in content,
            "CompetitionCSVManager import": "from utils.competition_csv_manager import CompetitionCSVManager" in content,
            "checkpoint_finder 초기화": "self.checkpoint_finder = CheckpointFinder()" in content,
            "csv_manager 초기화": "self.csv_manager = CompetitionCSVManager()" in content,
            "find_latest_checkpoint 사용": "self.checkpoint_finder.find_latest_checkpoint" in content
        }
        
        print("\n✅ auto_experiment_runner.py 수정 확인:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"   {check_name}: {status}")
            if not passed:
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("채점용 CSV 생성 시스템 통합 테스트")
    print("=" * 60)
    
    results = {
        "CheckpointFinder": test_checkpoint_finder(),
        "CompetitionCSVManager": test_competition_csv_manager(),
        "Integration": test_integration()
    }
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ 모든 테스트 통과! 시스템이 정상적으로 구현되었습니다.")
        print("\n다음 단계:")
        print("1. 1에포크 테스트: bash run_main_5_experiments.sh -1")
        print("2. 전체 실험 실행: bash run_main_5_experiments.sh")
    else:
        print("\n❌ 일부 테스트 실패. 위 오류를 확인하세요.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
