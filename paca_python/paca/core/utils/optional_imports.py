"""
Optional Imports Utility Module
선택적 의존성 import를 위한 유틸리티 모듈
"""

import importlib
import warnings
from typing import Optional, Any, Dict, Set


class OptionalImportError(ImportError):
    """선택적 import 실패 에러"""
    pass


class OptionalImports:
    """선택적 의존성 관리 클래스"""

    _available_modules: Dict[str, bool] = {}
    _import_warnings: Set[str] = set()

    @classmethod
    def is_available(cls, module_name: str) -> bool:
        """모듈이 사용 가능한지 확인"""
        if module_name not in cls._available_modules:
            try:
                importlib.import_module(module_name)
                cls._available_modules[module_name] = True
            except ImportError:
                cls._available_modules[module_name] = False
        return cls._available_modules[module_name]

    @classmethod
    def try_import(cls, module_name: str, package: Optional[str] = None) -> Optional[Any]:
        """모듈을 안전하게 import 시도"""
        try:
            return importlib.import_module(module_name, package)
        except ImportError:
            if module_name not in cls._import_warnings:
                warnings.warn(
                    f"Optional dependency '{module_name}' not found. "
                    f"Some features may not be available.",
                    category=UserWarning,
                    stacklevel=2
                )
                cls._import_warnings.add(module_name)
            return None

    @classmethod
    def require_module(cls, module_name: str, feature_name: str = None) -> Any:
        """필수 모듈을 import (실패 시 에러 발생)"""
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            feature_msg = f" for {feature_name}" if feature_name else ""
            raise OptionalImportError(
                f"Required module '{module_name}' not found{feature_msg}. "
                f"Please install it with: pip install -r requirements-optional.txt"
            ) from e


# 공통 선택적 의존성들
def try_import_torch():
    """PyTorch 선택적 import"""
    return OptionalImports.try_import('torch')

def try_import_transformers():
    """Transformers 선택적 import"""
    return OptionalImports.try_import('transformers')

def try_import_konlpy():
    """KoNLPy 선택적 import"""
    return OptionalImports.try_import('konlpy')

def try_import_pandas():
    """Pandas 선택적 import"""
    return OptionalImports.try_import('pandas')

def try_import_customtkinter():
    """CustomTkinter 선택적 import"""
    return OptionalImports.try_import('customtkinter')

def try_import_mqtt():
    """MQTT 선택적 import"""
    return OptionalImports.try_import('asyncio_mqtt')

def try_import_pystray():
    """PyStray 선택적 import"""
    return OptionalImports.try_import('pystray')

# 모니터링 및 로깅 관련 의존성들
def try_import_structlog():
    """structlog 선택적 import"""
    return OptionalImports.try_import('structlog')

def try_import_prometheus_client():
    """prometheus_client 선택적 import"""
    return OptionalImports.try_import('prometheus_client')

def try_import_psutil():
    """psutil 선택적 import"""
    return OptionalImports.try_import('psutil')

def try_import_uvloop():
    """uvloop 선택적 import (Unix only)"""
    return OptionalImports.try_import('uvloop')

def try_import_aiohttp():
    """aiohttp 선택적 import"""
    return OptionalImports.try_import('aiohttp')

def try_import_pydantic():
    """pydantic 선택적 import"""
    return OptionalImports.try_import('pydantic')


# 기능별 가용성 체크 함수들
def has_ai_features() -> bool:
    """AI/ML 기능 사용 가능 여부"""
    return (OptionalImports.is_available('transformers') and
            OptionalImports.is_available('torch'))

def has_korean_nlp() -> bool:
    """한국어 NLP 기능 사용 가능 여부"""
    return OptionalImports.is_available('konlpy')

def has_gui_features() -> bool:
    """GUI 기능 사용 가능 여부"""
    return OptionalImports.is_available('customtkinter')

def has_data_analysis() -> bool:
    """데이터 분석 기능 사용 가능 여부"""
    return OptionalImports.is_available('pandas')

def has_system_tray() -> bool:
    """시스템 트레이 기능 사용 가능 여부"""
    return OptionalImports.is_available('pystray')

def has_mqtt_support() -> bool:
    """MQTT 통신 기능 사용 가능 여부"""
    return OptionalImports.is_available('asyncio_mqtt')

def has_monitoring_features() -> bool:
    """모니터링 기능 사용 가능 여부"""
    return (OptionalImports.is_available('structlog') and
            OptionalImports.is_available('prometheus_client'))

def has_system_monitoring() -> bool:
    """시스템 모니터링 기능 사용 가능 여부"""
    return OptionalImports.is_available('psutil')

def has_advanced_networking() -> bool:
    """고급 네트워킹 기능 사용 가능 여부"""
    return OptionalImports.is_available('aiohttp')

def has_data_validation() -> bool:
    """데이터 검증 기능 사용 가능 여부"""
    return OptionalImports.is_available('pydantic')

def has_performance_loop() -> bool:
    """고성능 이벤트 루프 사용 가능 여부"""
    return OptionalImports.is_available('uvloop')


def get_feature_availability() -> Dict[str, bool]:
    """모든 기능의 사용 가능 여부 반환"""
    return {
        'ai_features': has_ai_features(),
        'korean_nlp': has_korean_nlp(),
        'gui_features': has_gui_features(),
        'data_analysis': has_data_analysis(),
        'system_tray': has_system_tray(),
        'mqtt_support': has_mqtt_support(),
        'monitoring_features': has_monitoring_features(),
        'system_monitoring': has_system_monitoring(),
        'advanced_networking': has_advanced_networking(),
        'data_validation': has_data_validation(),
        'performance_loop': has_performance_loop(),
    }


def print_feature_status():
    """기능별 사용 가능 상태 출력"""
    features = get_feature_availability()
    print("PACA v5 기능 사용 가능 상태:")
    print("=" * 40)

    for feature, available in features.items():
        status = "✓ 사용 가능" if available else "✗ 사용 불가"
        feature_name = {
            'ai_features': 'AI/ML 기능',
            'korean_nlp': '한국어 NLP',
            'gui_features': 'GUI 인터페이스',
            'data_analysis': '데이터 분석',
            'system_tray': '시스템 트레이',
            'mqtt_support': 'MQTT 통신'
        }.get(feature, feature)

        print(f"{feature_name:<15}: {status}")

    unavailable_features = [k for k, v in features.items() if not v]
    if unavailable_features:
        print("\n추가 기능을 사용하려면 다음 명령을 실행하세요:")
        print("pip install -r requirements-optional.txt")


# 편의 함수들
def safe_import(module_name: str, fallback=None):
    """안전한 import (실패 시 fallback 반환)"""
    module = OptionalImports.try_import(module_name)
    return module if module is not None else fallback


def require_feature(feature_name: str, modules: list):
    """기능 사용을 위한 필수 모듈들 확인"""
    missing_modules = []
    for module in modules:
        if not OptionalImports.is_available(module):
            missing_modules.append(module)

    if missing_modules:
        raise OptionalImportError(
            f"Feature '{feature_name}' requires the following modules: {missing_modules}. "
            f"Please install them with: pip install -r requirements-optional.txt"
        )


def check_dependencies_status() -> Dict[str, Any]:
    """의존성 상태 확인"""
    # 모든 기능 확인
    features = get_feature_availability()

    # 개별 모듈 확인
    modules = [
        'structlog', 'prometheus_client', 'psutil', 'uvloop',
        'aiohttp', 'pydantic', 'torch', 'transformers', 'konlpy',
        'pandas', 'customtkinter', 'asyncio_mqtt', 'pystray'
    ]

    available_modules = sum(1 for module in modules if OptionalImports.is_available(module))

    return {
        'total': len(modules),
        'available': available_modules,
        'missing': len(modules) - available_modules,
        'features': features,
        'modules': {module: OptionalImports.is_available(module) for module in modules}
    }