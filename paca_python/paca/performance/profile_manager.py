"""
Performance Profile Manager - PACA Python v5
4가지 성능 프로파일을 관리하고 동적으로 전환하는 시스템

LOW_END, MID_RANGE, HIGH_END, CONSERVATIVE 프로파일을 제공하며
시스템 상태에 따라 자동으로 최적의 프로파일을 선택
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id,
        create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id,
        create_success, create_failure
    )


class ProfileType(Enum):
    """성능 프로파일 타입"""
    LOW_END = "low-end"
    MID_RANGE = "mid-range"
    HIGH_END = "high-end"
    CONSERVATIVE = "conservative"


@dataclass
class ProfileConfig:
    """성능 프로파일 설정"""
    max_workers: int
    reasoning_steps: int
    speed_multiplier: float
    memory_limit_mb: int
    cache_size: int
    parallel_enabled: bool
    optimization_level: int  # 0-3
    timeout_seconds: float

    # 추가 설정
    enable_detailed_logging: bool = False
    enable_gc_optimization: bool = True
    enable_async_processing: bool = True
    batch_size: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileConfig':
        """딕셔너리에서 생성"""
        return cls(**data)

    @property
    def performance_score(self) -> float:
        """성능 점수 계산 (0-100)"""
        base_score = (
            self.max_workers * 10 +
            self.reasoning_steps * 5 +
            self.speed_multiplier * 20 +
            self.optimization_level * 15
        )
        return min(100, base_score)


@dataclass
class PerformanceProfile:
    """완전한 성능 프로파일"""
    profile_type: ProfileType
    name: str
    description: str
    config: ProfileConfig
    created_at: Timestamp = field(default_factory=current_timestamp)
    last_used: Optional[Timestamp] = None
    usage_count: int = 0

    def apply_profile(self) -> None:
        """프로파일 적용 (사용 통계 업데이트)"""
        self.last_used = current_timestamp()
        self.usage_count += 1

    @property
    def is_resource_intensive(self) -> bool:
        """리소스 집약적 프로파일 여부"""
        return (self.config.max_workers >= 6 or
                self.config.reasoning_steps >= 8 or
                self.config.memory_limit_mb >= 1000)


class ProfileManager:
    """
    성능 프로파일 관리자

    4가지 기본 프로파일을 제공하고 시스템 상태에 따라
    자동으로 최적의 프로파일을 선택하며 동적 전환을 지원
    """

    # 기본 프로파일 정의
    DEFAULT_PROFILES = {
        ProfileType.LOW_END: ProfileConfig(
            max_workers=2,
            reasoning_steps=3,
            speed_multiplier=0.5,
            memory_limit_mb=256,
            cache_size=50,
            parallel_enabled=False,
            optimization_level=0,
            timeout_seconds=30.0,
            enable_detailed_logging=False,
            enable_gc_optimization=True,
            enable_async_processing=False,
            batch_size=5
        ),

        ProfileType.MID_RANGE: ProfileConfig(
            max_workers=4,
            reasoning_steps=5,
            speed_multiplier=0.8,
            memory_limit_mb=512,
            cache_size=100,
            parallel_enabled=True,
            optimization_level=1,
            timeout_seconds=60.0,
            enable_detailed_logging=False,
            enable_gc_optimization=True,
            enable_async_processing=True,
            batch_size=10
        ),

        ProfileType.HIGH_END: ProfileConfig(
            max_workers=8,
            reasoning_steps=10,
            speed_multiplier=1.0,
            memory_limit_mb=1024,
            cache_size=200,
            parallel_enabled=True,
            optimization_level=2,
            timeout_seconds=120.0,
            enable_detailed_logging=True,
            enable_gc_optimization=True,
            enable_async_processing=True,
            batch_size=20
        ),

        ProfileType.CONSERVATIVE: ProfileConfig(
            max_workers=1,
            reasoning_steps=2,
            speed_multiplier=0.3,
            memory_limit_mb=128,
            cache_size=25,
            parallel_enabled=False,
            optimization_level=0,
            timeout_seconds=15.0,
            enable_detailed_logging=False,
            enable_gc_optimization=True,
            enable_async_processing=False,
            batch_size=3
        )
    }

    def __init__(self,
                 config_file: Optional[str] = None,
                 auto_switch: bool = True):
        """
        프로파일 매니저 초기화

        Args:
            config_file: 설정 파일 경로 (선택적)
            auto_switch: 자동 프로파일 전환 여부
        """
        self.config_file = config_file
        self.auto_switch = auto_switch

        # 기본 프로파일 생성
        self._profiles: Dict[ProfileType, PerformanceProfile] = {}
        self._initialize_default_profiles()

        self._current_profile_type = ProfileType.MID_RANGE
        self._switch_callbacks: List[Callable[[PerformanceProfile], None]] = []

        # 로깅 설정
        self.logger = logging.getLogger(__name__)

        # 설정 파일 로드
        if config_file:
            self._load_profiles_from_file()

    def _initialize_default_profiles(self) -> None:
        """기본 프로파일들 초기화"""
        profile_descriptions = {
            ProfileType.LOW_END: "저사양 시스템용 프로파일 - 최소한의 리소스 사용",
            ProfileType.MID_RANGE: "중간 사양용 프로파일 - 균형잡힌 성능과 효율성",
            ProfileType.HIGH_END: "고사양 시스템용 프로파일 - 최대 성능 추구",
            ProfileType.CONSERVATIVE: "안전 모드 프로파일 - 극도로 안정적인 실행"
        }

        for profile_type, config in self.DEFAULT_PROFILES.items():
            self._profiles[profile_type] = PerformanceProfile(
                profile_type=profile_type,
                name=profile_type.value,
                description=profile_descriptions[profile_type],
                config=config
            )

    @property
    def current_profile(self) -> PerformanceProfile:
        """현재 활성 프로파일"""
        return self._profiles[self._current_profile_type]

    @property
    def current_profile_type(self) -> ProfileType:
        """현재 프로파일 타입"""
        return self._current_profile_type

    @property
    def available_profiles(self) -> List[PerformanceProfile]:
        """사용 가능한 모든 프로파일"""
        return list(self._profiles.values())

    def get_profile(self, profile_type: ProfileType) -> Optional[PerformanceProfile]:
        """특정 프로파일 조회"""
        return self._profiles.get(profile_type)

    def add_switch_callback(self, callback: Callable[[PerformanceProfile], None]) -> None:
        """프로파일 전환 콜백 추가"""
        if callback not in self._switch_callbacks:
            self._switch_callbacks.append(callback)

    def remove_switch_callback(self, callback: Callable[[PerformanceProfile], None]) -> None:
        """프로파일 전환 콜백 제거"""
        if callback in self._switch_callbacks:
            self._switch_callbacks.remove(callback)

    def switch_profile(self, profile_type: ProfileType, reason: str = "") -> Result[PerformanceProfile]:
        """프로파일 전환"""
        try:
            if profile_type not in self._profiles:
                return create_failure(Exception(f"존재하지 않는 프로파일: {profile_type}"))

            if profile_type == self._current_profile_type:
                return create_success(self.current_profile)

            # 이전 프로파일 정보
            old_profile = self.current_profile

            # 프로파일 전환
            self._current_profile_type = profile_type
            new_profile = self.current_profile
            new_profile.apply_profile()

            # 로깅
            self.logger.info(f"프로파일 전환: {old_profile.name} → {new_profile.name}")
            if reason:
                self.logger.info(f"전환 이유: {reason}")

            # 콜백 호출
            for callback in self._switch_callbacks:
                try:
                    callback(new_profile)
                except Exception as e:
                    self.logger.warning(f"프로파일 전환 콜백 실패: {e}")

            return create_success(new_profile)

        except Exception as e:
            return create_failure(Exception(f"프로파일 전환 실패: {e}"))

    def auto_select_profile(self,
                          cpu_percent: float,
                          memory_percent: float,
                          available_memory_mb: float) -> Result[PerformanceProfile]:
        """
        시스템 상태 기반 자동 프로파일 선택

        Args:
            cpu_percent: CPU 사용률 (0-100)
            memory_percent: 메모리 사용률 (0-100)
            available_memory_mb: 사용 가능한 메모리 (MB)
        """
        try:
            # 선택 로직
            recommended_type = self._calculate_optimal_profile(
                cpu_percent, memory_percent, available_memory_mb
            )

            reason = f"시스템 상태 기반 자동 선택 (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"

            return self.switch_profile(recommended_type, reason)

        except Exception as e:
            return create_failure(Exception(f"자동 프로파일 선택 실패: {e}"))

    def _calculate_optimal_profile(self,
                                 cpu_percent: float,
                                 memory_percent: float,
                                 available_memory_mb: float) -> ProfileType:
        """최적 프로파일 계산"""

        # 극한 상황 - Conservative
        if (cpu_percent > 95 or memory_percent > 95 or available_memory_mb < 100):
            return ProfileType.CONSERVATIVE

        # 고부하 상황 - Low End
        if (cpu_percent > 80 or memory_percent > 85 or available_memory_mb < 300):
            return ProfileType.LOW_END

        # 중간 부하 상황 - Mid Range
        if (cpu_percent > 50 or memory_percent > 60 or available_memory_mb < 800):
            return ProfileType.MID_RANGE

        # 여유 상황 - High End
        return ProfileType.HIGH_END

    def customize_profile(self,
                         profile_type: ProfileType,
                         config_updates: Dict[str, Any]) -> Result[PerformanceProfile]:
        """프로파일 사용자 정의"""
        try:
            if profile_type not in self._profiles:
                return create_failure(Exception(f"존재하지 않는 프로파일: {profile_type}"))

            profile = self._profiles[profile_type]
            current_config = profile.config.to_dict()

            # 설정 업데이트
            current_config.update(config_updates)

            # 검증
            validated_config = self._validate_config(current_config)
            if not validated_config:
                return create_failure(Exception("잘못된 설정 값입니다"))

            # 새 설정 적용
            profile.config = ProfileConfig.from_dict(current_config)

            self.logger.info(f"프로파일 사용자 정의 완료: {profile_type.value}")

            return create_success(profile)

        except Exception as e:
            return create_failure(Exception(f"프로파일 사용자 정의 실패: {e}"))

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 값 검증"""
        try:
            # 필수 필드 확인
            required_fields = ['max_workers', 'reasoning_steps', 'speed_multiplier',
                             'memory_limit_mb', 'cache_size', 'timeout_seconds']

            for field in required_fields:
                if field not in config:
                    return False

            # 범위 검증
            if not (1 <= config['max_workers'] <= 16):
                return False
            if not (1 <= config['reasoning_steps'] <= 20):
                return False
            if not (0.1 <= config['speed_multiplier'] <= 2.0):
                return False
            if not (64 <= config['memory_limit_mb'] <= 4096):
                return False
            if not (10 <= config['cache_size'] <= 1000):
                return False
            if not (5.0 <= config['timeout_seconds'] <= 300.0):
                return False

            return True

        except Exception:
            return False

    def reset_to_defaults(self) -> Result[bool]:
        """모든 프로파일을 기본값으로 재설정"""
        try:
            self._initialize_default_profiles()
            self._current_profile_type = ProfileType.MID_RANGE

            self.logger.info("모든 프로파일이 기본값으로 재설정되었습니다")
            return create_success(True)

        except Exception as e:
            return create_failure(Exception(f"기본값 재설정 실패: {e}"))

    def get_performance_statistics(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        total_usage = sum(profile.usage_count for profile in self._profiles.values())

        stats = {
            'total_profile_switches': total_usage,
            'current_profile': self.current_profile.name,
            'current_performance_score': self.current_profile.config.performance_score,
            'profiles': {}
        }

        for profile_type, profile in self._profiles.items():
            usage_percentage = (profile.usage_count / total_usage * 100) if total_usage > 0 else 0

            stats['profiles'][profile_type.value] = {
                'usage_count': profile.usage_count,
                'usage_percentage': usage_percentage,
                'last_used': profile.last_used,
                'performance_score': profile.config.performance_score,
                'resource_intensive': profile.is_resource_intensive
            }

        return stats

    def _save_profiles_to_file(self) -> None:
        """프로파일을 파일로 저장"""
        if not self.config_file:
            return

        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'current_profile': self._current_profile_type.value,
                'profiles': {}
            }

            for profile_type, profile in self._profiles.items():
                data['profiles'][profile_type.value] = {
                    'name': profile.name,
                    'description': profile.description,
                    'config': profile.config.to_dict(),
                    'usage_count': profile.usage_count,
                    'last_used': profile.last_used
                }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"프로파일 설정 저장: {config_path}")

        except Exception as e:
            self.logger.error(f"프로파일 저장 실패: {e}")

    def _load_profiles_from_file(self) -> None:
        """파일에서 프로파일 로드"""
        if not self.config_file or not Path(self.config_file).exists():
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 현재 프로파일 설정
            if 'current_profile' in data:
                try:
                    self._current_profile_type = ProfileType(data['current_profile'])
                except ValueError:
                    pass  # 잘못된 값이면 기본값 유지

            # 프로파일 데이터 로드
            if 'profiles' in data:
                for profile_type_str, profile_data in data['profiles'].items():
                    try:
                        profile_type = ProfileType(profile_type_str)
                        if profile_type in self._profiles:
                            profile = self._profiles[profile_type]

                            # 설정 업데이트
                            if 'config' in profile_data:
                                profile.config = ProfileConfig.from_dict(profile_data['config'])

                            # 통계 정보 업데이트
                            if 'usage_count' in profile_data:
                                profile.usage_count = profile_data['usage_count']
                            if 'last_used' in profile_data:
                                profile.last_used = profile_data['last_used']

                    except ValueError:
                        continue  # 잘못된 프로파일 타입은 무시

            self.logger.info(f"프로파일 설정 로드: {self.config_file}")

        except Exception as e:
            self.logger.error(f"프로파일 로드 실패: {e}")

    def save_configuration(self) -> Result[bool]:
        """현재 설정을 파일로 저장"""
        try:
            self._save_profiles_to_file()
            return create_success(True)
        except Exception as e:
            return create_failure(Exception(f"설정 저장 실패: {e}"))

    def recommend_profile(self,
                         workload_type: str,
                         expected_duration_minutes: float = 10.0) -> PerformanceProfile:
        """
        작업 유형과 예상 지속 시간 기반 프로파일 추천

        Args:
            workload_type: 작업 유형 ('light', 'moderate', 'heavy', 'critical')
            expected_duration_minutes: 예상 지속 시간 (분)
        """
        recommendations = {
            'light': ProfileType.LOW_END,
            'moderate': ProfileType.MID_RANGE,
            'heavy': ProfileType.HIGH_END,
            'critical': ProfileType.CONSERVATIVE
        }

        base_recommendation = recommendations.get(workload_type, ProfileType.MID_RANGE)

        # 지속 시간이 긴 경우 더 안정적인 프로파일 선택
        if expected_duration_minutes > 60:
            if base_recommendation == ProfileType.HIGH_END:
                base_recommendation = ProfileType.MID_RANGE
            elif base_recommendation == ProfileType.MID_RANGE:
                base_recommendation = ProfileType.LOW_END

        return self._profiles[base_recommendation]


# 편의 함수들
def create_custom_profile(name: str,
                         max_workers: int = 4,
                         reasoning_steps: int = 5,
                         speed_multiplier: float = 1.0) -> PerformanceProfile:
    """사용자 정의 프로파일 생성"""
    config = ProfileConfig(
        max_workers=max_workers,
        reasoning_steps=reasoning_steps,
        speed_multiplier=speed_multiplier,
        memory_limit_mb=512,
        cache_size=100,
        parallel_enabled=max_workers > 1,
        optimization_level=1,
        timeout_seconds=60.0
    )

    return PerformanceProfile(
        profile_type=ProfileType.MID_RANGE,  # 기본값
        name=name,
        description=f"사용자 정의 프로파일: {name}",
        config=config
    )


def get_profile_for_system_specs(cpu_cores: int,
                                total_memory_gb: float) -> ProfileType:
    """시스템 사양 기반 추천 프로파일"""
    if cpu_cores >= 8 and total_memory_gb >= 16:
        return ProfileType.HIGH_END
    elif cpu_cores >= 4 and total_memory_gb >= 8:
        return ProfileType.MID_RANGE
    elif cpu_cores >= 2 and total_memory_gb >= 4:
        return ProfileType.LOW_END
    else:
        return ProfileType.CONSERVATIVE


if __name__ == "__main__":
    # 테스트 실행
    async def main():
        print("=== PACA v5 성능 프로파일 매니저 테스트 ===")

        manager = ProfileManager()

        # 현재 프로파일 정보
        current = manager.current_profile
        print(f"현재 프로파일: {current.name}")
        print(f"설정: workers={current.config.max_workers}, "
              f"steps={current.config.reasoning_steps}, "
              f"speed={current.config.speed_multiplier}")
        print(f"성능 점수: {current.config.performance_score:.1f}")

        # 프로파일 전환 테스트
        print("\n프로파일 전환 테스트...")

        for profile_type in [ProfileType.HIGH_END, ProfileType.CONSERVATIVE, ProfileType.LOW_END]:
            result = manager.switch_profile(profile_type, "테스트 전환")
            if result.is_success:
                profile = result.value
                print(f"→ {profile.name}: {profile.config.performance_score:.1f}점")

        # 자동 선택 테스트
        print("\n자동 프로파일 선택 테스트...")
        test_cases = [
            (30, 40, 1000),  # 여유로운 상황
            (70, 60, 500),   # 중간 부하
            (90, 80, 200),   # 고부하
            (98, 95, 50)     # 임계 상황
        ]

        for cpu, memory, available_mb in test_cases:
            result = manager.auto_select_profile(cpu, memory, available_mb)
            if result.is_success:
                profile = result.value
                print(f"CPU {cpu}%, Memory {memory}% → {profile.name}")

        # 성능 통계
        print("\n성능 통계:")
        stats = manager.get_performance_statistics()
        for profile_name, profile_stats in stats['profiles'].items():
            print(f"{profile_name}: {profile_stats['usage_count']}회 사용")

        print("테스트 완료!")

    asyncio.run(main())