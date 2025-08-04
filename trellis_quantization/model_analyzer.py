"""
TRELLIS 모델 구조 분석 클래스

기능:
- 파이프라인 구조 분석
- 컴포넌트별 파라미터 계산
- 양자화 가능 여부 확인
"""

from typing import List, Tuple, Any
import torch.nn as nn


class ModelAnalyzer:
    """TRELLIS 모델 구조 분석기"""
    
    def analyze_pipeline(self, pipeline) -> List[Tuple[str, nn.Module]]:
        """
        파이프라인 구조 분석 및 컴포넌트 추출
        
        Args:
            pipeline: TRELLIS 파이프라인 객체
            
        Returns:
            List[Tuple[str, nn.Module]]: (컴포넌트명, 모듈) 리스트
        """
        print("📋 모델 구조 분석 중...")
        
        components = []
        total_params = 0
        
        # 1. models 딕셔너리 분석 (TRELLIS 주요 컴포넌트)
        if hasattr(pipeline, 'models') and isinstance(pipeline.models, dict):
            print("  📦 models 딕셔너리 분석:")
            
            for model_name, model in pipeline.models.items():
                if model is not None:
                    try:
                        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        total_params += model_params
                        components.append((model_name, model))
                        
                        # 양자화 상태 확인
                        is_quantized = self._check_quantization_status(model)
                        status = "🔧INT8" if is_quantized else "📏FP32"
                        
                        print(f"    • {model_name}: {model_params/1e6:.1f}M 파라미터 {status}")
                        
                    except Exception as e:
                        print(f"    ⚠️ {model_name}: 분석 실패 ({e})")
                else:
                    print(f"    ⚠️ {model_name}: None (로드되지 않음)")
        
        # 2. 추가 컴포넌트 분석 (text_encoder 등)
        additional_components = ['text_encoder', 'text_model']
        for attr_name in additional_components:
            if hasattr(pipeline, attr_name):
                attr_obj = getattr(pipeline, attr_name)
                if attr_obj is not None and hasattr(attr_obj, 'parameters'):
                    try:
                        attr_params = sum(p.numel() for p in attr_obj.parameters() if p.requires_grad)
                        if attr_params > 0:
                            total_params += attr_params
                            components.append((attr_name, attr_obj))
                            print(f"  📝 {attr_name}: {attr_params/1e6:.1f}M 파라미터")
                    except Exception as e:
                        print(f"  ⚠️ {attr_name}: 분석 실패 ({e})")
        
        # 3. tokenizer는 파라미터가 없으므로 정보만 출력
        if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
            try:
                vocab_size = len(pipeline.tokenizer.get_vocab())
                print(f"  📚 tokenizer: 어휘 크기 {vocab_size:,}")
            except:
                print(f"  📚 tokenizer: 존재 (파라미터 없음)")
        
        print(f"  📊 총 파라미터: {total_params/1e6:.1f}M")
        print(f"  🔧 양자화 대상: {len(components)}개 컴포넌트")
        
        return components
    
    def _check_quantization_status(self, module: nn.Module) -> bool:
        """
        모듈의 양자화 상태 확인
        
        Args:
            module: 확인할 모듈
            
        Returns:
            bool: 양자화 여부
        """
        try:
            # 양자화된 모듈들의 특징 확인
            for m in module.modules():
                # 1. _packed_params 속성 확인 (동적 양자화)
                if hasattr(m, '_packed_params'):
                    return True
                
                # 2. 클래스명에 'quantized' 포함 확인
                if 'quantized' in str(type(m)).lower():
                    return True
                
                # 3. qint8 타입 파라미터 확인
                for param in m.parameters():
                    if hasattr(param, 'dtype') and 'qint' in str(param.dtype):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def get_model_info(self, pipeline) -> dict:
        """
        파이프라인의 전체 정보 수집
        
        Args:
            pipeline: TRELLIS 파이프라인
            
        Returns:
            dict: 모델 정보
        """
        components = self.analyze_pipeline(pipeline)
        
        info = {
            'total_components': len(components),
            'total_parameters': 0,
            'component_details': {}
        }
        
        for name, module in components:
            try:
                params = sum(p.numel() for p in module.parameters())
                size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
                
                info['total_parameters'] += params
                info['component_details'][name] = {
                    'parameters': params,
                    'size_mb': size_bytes / (1024 * 1024),
                    'quantized': self._check_quantization_status(module)
                }
            except Exception as e:
                info['component_details'][name] = {
                    'error': str(e)
                }
        
        return info