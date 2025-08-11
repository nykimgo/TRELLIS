"""
TRELLIS 논문 방식의 렌더링 유틸리티

기존 dataset_toolkits/render.py를 활용하되, 
TRELLIS 논문 방법론에 맞는 8개 고정 뷰 렌더링을 제공합니다.
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# TRELLIS dataset_toolkits 경로 추가
TRELLIS_ROOT = Path(__file__).parent.parent
DATASET_TOOLKITS_PATH = TRELLIS_ROOT / "dataset_toolkits"
sys.path.insert(0, str(DATASET_TOOLKITS_PATH))

# TRELLIS 렌더링 모듈 import
try:
    from render import BLENDER_PATH, BLENDER_INSTALLATION_PATH, _install_blender
except ImportError as e:
    print(f"⚠️ TRELLIS 렌더링 모듈 import 실패: {e}")
    BLENDER_PATH = None


def create_trellis_paper_views() -> List[Dict]:
    """
    TRELLIS 논문 방식의 8개 카메라 뷰 생성
    
    논문 설정:
    - Yaw: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° (45도 간격)
    - Pitch: 30° (고정)
    - Radius: 2 (고정)
    - FOV: 40도 (기본값)
    
    Returns:
        List[Dict]: Blender 스크립트용 뷰 설정들
    """
    views = []
    
    # TRELLIS 논문 설정
    yaw_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]  # 45도 간격
    pitch_deg = 30  # 고정
    radius = 2  # 고정
    fov_deg = 40  # 기본값
    
    for yaw_deg in yaw_angles_deg:
        # 각도를 라디안으로 변환
        yaw_rad = yaw_deg / 180 * np.pi
        pitch_rad = pitch_deg / 180 * np.pi
        fov_rad = fov_deg / 180 * np.pi
        
        view = {
            'yaw': yaw_rad,
            'pitch': pitch_rad,
            'radius': radius,
            'fov': fov_rad
        }
        views.append(view)
    
    print(f"📷 TRELLIS 논문 방식: 8개 고정 카메라 뷰 생성")
    print(f"   Yaw: {yaw_angles_deg} (45도 간격)")
    print(f"   Pitch: {pitch_deg}° (고정)")
    print(f"   Radius: {radius} (고정)")
    
    return views


def render_3d_asset_trellis_paper(asset_file_path: str, output_dir: str, 
                                 asset_id: str = None) -> List[str]:
    """
    TRELLIS 논문 방식으로 3D 자산 렌더링
    
    Args:
        asset_file_path: 3D 자산 파일 경로 (.obj, .glb, .blend 등)
        output_dir: 렌더링 결과 저장 디렉토리
        asset_id: 자산 ID (None이면 자동 생성)
        
    Returns:
        List[str]: 렌더링된 이미지 파일 경로들
    """
    if not os.path.exists(asset_file_path):
        raise FileNotFoundError(f"3D 자산 파일을 찾을 수 없습니다: {asset_file_path}")
    
    # Blender 설치 확인
    if not os.path.exists(BLENDER_PATH):
        print("🔧 Blender 설치 중...")
        _install_blender()
    
    # 출력 디렉토리 설정
    if asset_id is None:
        asset_id = Path(asset_file_path).stem
    
    output_path = Path(output_dir)
    renders_dir = output_path / "renders" / asset_id
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    # TRELLIS 논문 방식 카메라 뷰 생성
    views = create_trellis_paper_views()
    
    # Blender 스크립트 경로
    blender_script_path = DATASET_TOOLKITS_PATH / "blender_script" / "render.py"
    
    if not blender_script_path.exists():
        raise FileNotFoundError(f"Blender 렌더링 스크립트를 찾을 수 없습니다: {blender_script_path}")
    
    # Blender 렌더링 명령 구성
    args = [
        BLENDER_PATH, '-b', '-P', str(blender_script_path),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(asset_file_path),
        '--resolution', '512',
        '--output_folder', str(renders_dir),
        '--engine', 'CYCLES',
    ]
    
    # .blend 파일인 경우 특별 처리
    if asset_file_path.endswith('.blend'):
        args.insert(1, asset_file_path)
    
    print(f"🎬 TRELLIS 논문 방식 렌더링 시작")
    print(f"   입력: {asset_file_path}")
    print(f"   출력: {renders_dir}")
    print(f"   뷰 수: 8개 (45도 간격)")
    
    # 렌더링 실행
    try:
        result = subprocess.run(
            args, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode != 0:
            print(f"❌ Blender 렌더링 실패:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return []
            
    except subprocess.TimeoutExpired:
        print("❌ 렌더링 시간 초과 (5분)")
        return []
    except Exception as e:
        print(f"❌ 렌더링 중 오류 발생: {e}")
        return []
    
    # 렌더링된 이미지 파일 찾기
    rendered_images = []
    
    # transforms.json이 있으면 렌더링 성공
    transforms_file = renders_dir / "transforms.json"
    if transforms_file.exists():
        # 8개 뷰에 해당하는 이미지 찾기
        for i in range(8):
            image_file = renders_dir / f"{i:04d}.png"
            if image_file.exists():
                rendered_images.append(str(image_file))
        
        print(f"✅ 렌더링 완료: {len(rendered_images)}/8개 이미지")
        
        if len(rendered_images) < 8:
            print(f"⚠️ 일부 이미지만 생성됨: {len(rendered_images)}/8")
    else:
        print("❌ 렌더링 실패: transforms.json 파일이 생성되지 않음")
    
    return rendered_images


def load_rendered_images(image_paths: List[str]) -> List[Image.Image]:
    """렌더링된 이미지들을 PIL Image로 로드"""
    images = []
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"⚠️ 이미지 로드 실패 {path}: {e}")
    
    print(f"📸 로드된 이미지: {len(images)}개")
    return images


def render_pipeline_output_trellis_paper(pipeline_output: Any, output_dir: str, 
                                        asset_id: str = None) -> List[str]:
    """
    TRELLIS 파이프라인 출력을 TRELLIS 논문 방식으로 렌더링
    
    Args:
        pipeline_output: TRELLIS 파이프라인 출력 객체
        output_dir: 렌더링 결과 저장 디렉토리  
        asset_id: 자산 ID
        
    Returns:
        List[str]: 렌더링된 이미지 파일 경로들
    """
    # 임시 파일로 3D 자산 저장
    temp_dir = tempfile.mkdtemp()
    
    try:
        # pipeline_output을 파일로 저장하는 로직
        if hasattr(pipeline_output, 'save'):
            # 메쉬나 가우시안 스플래팅 등을 파일로 저장
            temp_asset_path = os.path.join(temp_dir, "generated_asset.obj")
            pipeline_output.save(temp_asset_path)
        elif hasattr(pipeline_output, 'export'):
            temp_asset_path = os.path.join(temp_dir, "generated_asset.glb")
            pipeline_output.export(temp_asset_path)
        else:
            # 기본적으로 .obj 형식으로 저장 시도
            temp_asset_path = os.path.join(temp_dir, "generated_asset.obj")
            
            # 실제 저장 로직은 pipeline_output의 구조에 따라 구현 필요
            # 현재는 더미 OBJ 파일 생성
            create_dummy_obj_file(temp_asset_path)
            print("⚠️ 파이프라인 출력을 파일로 저장하는 로직이 필요합니다. 더미 파일 사용.")
        
        # 저장된 파일을 렌더링
        return render_3d_asset_trellis_paper(temp_asset_path, output_dir, asset_id)
        
    except Exception as e:
        print(f"❌ 파이프라인 출력 렌더링 실패: {e}")
        return []
    finally:
        # 임시 파일 정리
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def create_dummy_obj_file(output_path: str):
    """테스트용 더미 OBJ 파일 생성"""
    obj_content = """
# TRELLIS 테스트용 더미 큐브
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""
    
    with open(output_path, 'w') as f:
        f.write(obj_content)


def test_rendering():
    """렌더링 시스템 테스트"""
    print("🧪 TRELLIS 렌더링 시스템 테스트")
    
    # 테스트용 더미 OBJ 파일 생성
    test_dir = Path("/tmp/trellis_render_test")
    test_dir.mkdir(exist_ok=True)
    
    test_obj = test_dir / "test_cube.obj"
    create_dummy_obj_file(str(test_obj))
    
    print(f"🎲 테스트 파일 생성: {test_obj}")
    
    # TRELLIS 논문 방식 렌더링 테스트
    try:
        rendered_images = render_3d_asset_trellis_paper(
            str(test_obj), 
            str(test_dir), 
            "test_cube"
        )
        
        if rendered_images:
            images = load_rendered_images(rendered_images)
            print(f"✅ 테스트 성공: {len(images)}개 이미지 로드됨")
        else:
            print("❌ 렌더링 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    test_rendering()