#!/usr/bin/env python3
"""
auto_experiment_runner.py 패치 스크립트

체크포인트 탐색 로직만 수정하는 간단한 패치
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 필요한 모듈 가져오기
from utils.checkpoint_finder import CheckpointFinder
from utils.competition_csv_manager import CompetitionCSVManager

def patch_auto_experiment_runner():
    """auto_experiment_runner.py 패치"""
    file_path = project_root / "code" / "auto_experiment_runner.py"
    
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. import 추가
    import_line = "from utils import load_config"
    new_imports = """from utils import load_config
# 🆕 추가: 체크포인트 탐색기와 CSV 관리자
from utils.checkpoint_finder import CheckpointFinder
from utils.competition_csv_manager import CompetitionCSVManager"""
    
    content = content.replace(import_line, new_imports, 1)
    
    # 2. 초기화 추가
    init_line = '        self.csv_saver = CSVResultsSaver(f"{output_dir}/csv_results")'
    new_init = """        self.csv_saver = CSVResultsSaver(f"{output_dir}/csv_results")
        
        # 🆕 새로 추가: 체크포인트 탐색기와 CSV 관리자
        self.checkpoint_finder = CheckpointFinder()
        self.csv_manager = CompetitionCSVManager()"""
    
    content = content.replace(init_line, new_init, 1)
    
    # 3. 체크포인트 탐색 로직 수정
    old_checkpoint_logic = """                    # 베스트 체크포인트 찾기
                    output_dir = Path(config.get('training', {}).get('output_dir', 'outputs'))
                    checkpoint_dirs = list(output_dir.glob('checkpoint-*'))"""
    
    new_checkpoint_logic = """                    # 🔧 수정: 정확한 체크포인트 탐색
                    best_checkpoint = self.checkpoint_finder.find_latest_checkpoint(experiment_id)"""
    
    content = content.replace(old_checkpoint_logic, new_checkpoint_logic, 1)
    
    # 4. checkpoint_dirs 조건문 수정
    old_condition = "if checkpoint_dirs:"
    new_condition = "if best_checkpoint and self.checkpoint_finder.validate_checkpoint(best_checkpoint):"
    
    content = content.replace(old_condition, new_condition, 1)
    
    # 5. checkpoint 선택 로직 제거
    old_selection = """                        # 가장 최근 체크포인트 선택
                        best_checkpoint = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)"""
    content = content.replace(old_selection, "", 1)
    
    # 6. 체크포인트 없을 때 메시지 수정
    old_msg = "⚠️ 체크포인트를 찾을 수 없습니다."
    new_msg = "❌ 유효한 체크포인트를 찾을 수 없습니다."
    content = content.replace(old_msg, new_msg, 1)
    
    # 7. 추론 후 CSV 저장 추가
    # post_training_inference 성공 후
    old_success = """                            print(f"✅ 제출 파일 생성 완료: {submission_path}")
                            result = self._collect_results(config, Path(config_path).stem)
                            result['submission_path'] = submission_path"""
    
    new_success = """                            print(f"✅ 제출 파일 생성 완료: {submission_path}")
                            
                            # 🆕 채점용 CSV 생성
                            try:
                                import pandas as pd
                                result_df = pd.read_csv(submission_path)
                                competition_paths = self.csv_manager.save_experiment_submission(
                                    experiment_name=experiment_id,
                                    result_df=result_df,
                                    config=config,
                                    metrics=None  # 메트릭은 나중에 추가
                                )
                                result = self._collect_results(config, Path(config_path).stem)
                                result['submission_path'] = submission_path
                                result['competition_paths'] = competition_paths
                            except Exception as csv_e:
                                print(f"⚠️ 채점용 CSV 생성 실패: {csv_e}")
                                result = self._collect_results(config, Path(config_path).stem)
                                result['submission_path'] = submission_path"""
    
    content = content.replace(old_success, new_success, 1)
    
    # 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ auto_experiment_runner.py 패치 완료!")
    print("   - CheckpointFinder 통합")
    print("   - CompetitionCSVManager 통합")
    print("   - 체크포인트 탐색 로직 수정")
    print("   - 채점용 CSV 자동 생성 추가")

if __name__ == "__main__":
    patch_auto_experiment_runner()
