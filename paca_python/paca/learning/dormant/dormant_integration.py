"""
휴면기 통합 시스템 메인 클래스

시스템이 활발하게 사용되지 않는 휴면 상태에서 메모리 정리와 패턴 강화를 수행합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from .memory_consolidator import MemoryConsolidator
from .pattern_strengthener import PatternStrengthener
from .weak_connection_pruner import WeakConnectionPruner


class DormantPhase(Enum):
    """휴면기 단계"""
    INACTIVE = "inactive"
    ANALYZING = "analyzing"
    CONSOLIDATING = "consolidating"
    STRENGTHENING = "strengthening"
    PRUNING = "pruning"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class DormantConfig:
    """휴면기 설정"""
    min_dormant_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    max_processing_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    memory_threshold: int = 1000  # 처리할 최소 메모리 항목 수
    pattern_confidence_threshold: float = 0.7
    weak_connection_threshold: float = 0.3
    consolidation_batch_size: int = 100
    enable_parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class DormantSession:
    """휴면기 세션"""
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    phase: DormantPhase = DormantPhase.INACTIVE
    progress: float = 0.0
    processed_memories: int = 0
    strengthened_patterns: int = 0
    pruned_connections: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'phase': self.phase.value,
            'progress': self.progress,
            'processed_memories': self.processed_memories,
            'strengthened_patterns': self.strengthened_patterns,
            'pruned_connections': self.pruned_connections,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class DormantResult:
    """휴면기 처리 결과"""
    session_id: str
    success: bool
    duration: timedelta
    total_processed: int
    memory_consolidation: Dict[str, Any]
    pattern_strengthening: Dict[str, Any]
    connection_pruning: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'success': self.success,
            'duration_seconds': self.duration.total_seconds(),
            'total_processed': self.total_processed,
            'memory_consolidation': self.memory_consolidation,
            'pattern_strengthening': self.pattern_strengthening,
            'connection_pruning': self.connection_pruning,
            'performance_metrics': self.performance_metrics,
            'recommendations': self.recommendations
        }


class DormantIntegration:
    """휴면기 통합 시스템 메인 클래스"""

    def __init__(self, config: Optional[DormantConfig] = None):
        self.config = config or DormantConfig()
        self.logger = logging.getLogger(__name__)

        # 구성 요소 초기화
        self.memory_consolidator = MemoryConsolidator()
        self.pattern_strengthener = PatternStrengthener()
        self.connection_pruner = WeakConnectionPruner()

        # 세션 관리
        self.current_session: Optional[DormantSession] = None
        self.session_history: List[DormantSession] = []

        # 상태 관리
        self.last_activity_time: Optional[datetime] = None
        self.is_processing: bool = False

        # 성능 모니터링
        self.performance_stats = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'total_memories_processed': 0,
            'total_patterns_strengthened': 0,
            'total_connections_pruned': 0
        }

    async def start_dormant_processing(self) -> DormantSession:
        """휴면기 처리 시작"""
        if self.is_processing:
            raise RuntimeError("이미 휴면기 처리가 진행 중입니다.")

        # 새 세션 생성
        session = DormantSession(
            id=f"dormant_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )

        self.current_session = session
        self.is_processing = True

        self.logger.info(f"휴면기 처리 시작: {session.id}")

        try:
            # 처리 단계별 실행
            await self._run_dormant_phases(session)

            session.phase = DormantPhase.COMPLETED
            session.end_time = datetime.now()

            self.logger.info(f"휴면기 처리 완료: {session.id}")

        except Exception as e:
            session.phase = DormantPhase.ERROR
            session.error = str(e)
            session.end_time = datetime.now()

            self.logger.error(f"휴면기 처리 오류: {session.id} - {str(e)}")

        finally:
            self.is_processing = False
            self.session_history.append(session)
            self.current_session = None

            # 통계 업데이트
            self._update_performance_stats(session)

        return session

    async def _run_dormant_phases(self, session: DormantSession):
        """휴면기 처리 단계별 실행"""
        start_time = datetime.now()
        timeout = self.config.max_processing_time

        try:
            # Phase 1: 분석 단계
            session.phase = DormantPhase.ANALYZING
            session.progress = 0.1

            analysis_result = await asyncio.wait_for(
                self._analyze_memory_state(),
                timeout=timeout.total_seconds() / 4
            )

            session.metadata['analysis'] = analysis_result
            session.progress = 0.3

            # Phase 2: 메모리 통합
            session.phase = DormantPhase.CONSOLIDATING

            consolidation_result = await asyncio.wait_for(
                self._consolidate_memories(analysis_result),
                timeout=timeout.total_seconds() / 3
            )

            session.processed_memories = consolidation_result['processed_count']
            session.progress = 0.6

            # Phase 3: 패턴 강화
            session.phase = DormantPhase.STRENGTHENING

            strengthening_result = await asyncio.wait_for(
                self._strengthen_patterns(analysis_result),
                timeout=timeout.total_seconds() / 3
            )

            session.strengthened_patterns = strengthening_result['strengthened_count']
            session.progress = 0.8

            # Phase 4: 약한 연결 정리
            session.phase = DormantPhase.PRUNING

            pruning_result = await asyncio.wait_for(
                self._prune_weak_connections(analysis_result),
                timeout=timeout.total_seconds() / 4
            )

            session.pruned_connections = pruning_result['pruned_count']
            session.progress = 1.0

            # 최종 결과 저장
            session.metadata.update({
                'consolidation': consolidation_result,
                'strengthening': strengthening_result,
                'pruning': pruning_result
            })

        except asyncio.TimeoutError:
            raise RuntimeError(f"휴면기 처리 시간 초과: {timeout}")
        except Exception as e:
            raise RuntimeError(f"휴면기 처리 단계 오류: {str(e)}")

    async def _analyze_memory_state(self) -> Dict[str, Any]:
        """메모리 상태 분석"""
        self.logger.info("메모리 상태 분석 시작")

        # 시뮬레이션된 메모리 분석
        # 실제 구현에서는 실제 메모리 시스템과 연동

        analysis = {
            'total_memories': 5000,
            'recent_memories': 1200,
            'old_memories': 3800,
            'weak_connections': 800,
            'strong_patterns': 150,
            'memory_fragmentation': 0.3,
            'pattern_strength_distribution': {
                'very_weak': 400,
                'weak': 300,
                'medium': 200,
                'strong': 80,
                'very_strong': 20
            },
            'recommended_actions': [
                'consolidate_old_memories',
                'strengthen_medium_patterns',
                'prune_very_weak_connections'
            ]
        }

        await asyncio.sleep(0.5)  # 분석 시뮬레이션

        self.logger.info(f"메모리 분석 완료: {analysis['total_memories']}개 메모리 분석됨")
        return analysis

    async def _consolidate_memories(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """메모리 통합 실행"""
        self.logger.info("메모리 통합 시작")

        # 메모리 통합 처리
        result = await self.memory_consolidator.consolidate_memories(
            memory_count=analysis['total_memories'],
            batch_size=self.config.consolidation_batch_size,
            parallel=self.config.enable_parallel_processing
        )

        self.logger.info(f"메모리 통합 완료: {result['processed_count']}개 처리됨")
        return result

    async def _strengthen_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 강화 실행"""
        self.logger.info("패턴 강화 시작")

        # 패턴 강화 처리
        result = await self.pattern_strengthener.strengthen_patterns(
            pattern_data=analysis['pattern_strength_distribution'],
            confidence_threshold=self.config.pattern_confidence_threshold
        )

        self.logger.info(f"패턴 강화 완료: {result['strengthened_count']}개 강화됨")
        return result

    async def _prune_weak_connections(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """약한 연결 정리 실행"""
        self.logger.info("약한 연결 정리 시작")

        # 약한 연결 정리 처리
        result = await self.connection_pruner.prune_connections(
            weak_connections=analysis['weak_connections'],
            threshold=self.config.weak_connection_threshold
        )

        self.logger.info(f"약한 연결 정리 완료: {result['pruned_count']}개 제거됨")
        return result

    def should_start_dormant_processing(self) -> bool:
        """휴면기 처리 시작 여부 판단"""
        if self.is_processing:
            return False

        if self.last_activity_time is None:
            return False

        inactive_duration = datetime.now() - self.last_activity_time
        return inactive_duration >= self.config.min_dormant_duration

    def mark_activity(self):
        """활동 시점 기록"""
        self.last_activity_time = datetime.now()

    async def schedule_dormant_processing(self, interval_hours: int = 4):
        """정기적 휴면기 처리 스케줄링"""
        self.logger.info(f"휴면기 처리 스케줄링 시작: {interval_hours}시간 간격")

        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # 시간 단위를 초로 변환

                if self.should_start_dormant_processing():
                    self.logger.info("정기 휴면기 처리 시작")
                    session = await self.start_dormant_processing()

                    if session.phase == DormantPhase.ERROR:
                        self.logger.error(f"정기 휴면기 처리 실패: {session.error}")
                    else:
                        self.logger.info(f"정기 휴면기 처리 성공: {session.id}")

            except Exception as e:
                self.logger.error(f"휴면기 스케줄링 오류: {str(e)}")
                await asyncio.sleep(300)  # 5분 대기 후 재시도

    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            'is_processing': self.is_processing,
            'current_phase': self.current_session.phase.value if self.current_session else None,
            'progress': self.current_session.progress if self.current_session else 0.0,
            'last_activity': self.last_activity_time.isoformat() if self.last_activity_time else None,
            'session_count': len(self.session_history),
            'performance_stats': self.performance_stats.copy()
        }

    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """세션 이력 조회"""
        recent_sessions = self.session_history[-limit:] if limit > 0 else self.session_history
        return [session.to_dict() for session in recent_sessions]

    def get_recommendations(self) -> List[str]:
        """시스템 개선 권장사항 제공"""
        recommendations = []

        # 성능 기반 권장사항
        if self.performance_stats['total_sessions'] > 0:
            success_rate = self.performance_stats['successful_sessions'] / self.performance_stats['total_sessions']

            if success_rate < 0.8:
                recommendations.append("휴면기 처리 성공률이 낮습니다. 설정을 조정하세요.")

            if self.performance_stats['average_processing_time'] > 1800:  # 30분 초과
                recommendations.append("평균 처리 시간이 깁니다. 병렬 처리를 활성화하거나 배치 크기를 줄이세요.")

        # 메모리 기반 권장사항
        if hasattr(self, 'last_analysis') and self.last_analysis:
            fragmentation = self.last_analysis.get('memory_fragmentation', 0)
            if fragmentation > 0.5:
                recommendations.append("메모리 단편화가 심합니다. 통합 주기를 단축하세요.")

        return recommendations

    def _update_performance_stats(self, session: DormantSession):
        """성능 통계 업데이트"""
        self.performance_stats['total_sessions'] += 1

        if session.phase == DormantPhase.COMPLETED:
            self.performance_stats['successful_sessions'] += 1

        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
            self.performance_stats['total_processing_time'] += duration
            self.performance_stats['average_processing_time'] = (
                self.performance_stats['total_processing_time'] /
                self.performance_stats['total_sessions']
            )

        self.performance_stats['total_memories_processed'] += session.processed_memories
        self.performance_stats['total_patterns_strengthened'] += session.strengthened_patterns
        self.performance_stats['total_connections_pruned'] += session.pruned_connections

    async def cleanup_old_sessions(self, days: int = 30):
        """오래된 세션 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)

        original_count = len(self.session_history)
        self.session_history = [
            session for session in self.session_history
            if session.start_time > cutoff_date
        ]

        cleaned_count = original_count - len(self.session_history)

        if cleaned_count > 0:
            self.logger.info(f"오래된 세션 {cleaned_count}개 정리됨")

        return cleaned_count

    async def export_session_data(self, session_ids: List[str] = None) -> Dict[str, Any]:
        """세션 데이터 내보내기"""
        if session_ids is None:
            sessions_to_export = self.session_history
        else:
            sessions_to_export = [
                session for session in self.session_history
                if session.id in session_ids
            ]

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_sessions': len(sessions_to_export),
            'performance_summary': self.performance_stats.copy(),
            'sessions': [session.to_dict() for session in sessions_to_export]
        }

        return export_data