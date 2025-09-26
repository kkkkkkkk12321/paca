"""
약한 연결 정리기 (Weak Connection Pruner)

약한 연결 제거 및 메모리 네트워크 정리를 담당합니다.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import math


class ConnectionType(Enum):
    """연결 유형"""
    SEMANTIC = "semantic"       # 의미적 연결
    TEMPORAL = "temporal"       # 시간적 연결
    CAUSAL = "causal"          # 인과적 연결
    ASSOCIATIVE = "associative" # 연상적 연결
    HIERARCHICAL = "hierarchical" # 계층적 연결


class ConnectionStrength(Enum):
    """연결 강도"""
    VERY_WEAK = "very_weak"     # 0.0 - 0.2
    WEAK = "weak"               # 0.2 - 0.4
    MEDIUM = "medium"           # 0.4 - 0.6
    STRONG = "strong"           # 0.6 - 0.8
    VERY_STRONG = "very_strong" # 0.8 - 1.0


@dataclass
class Connection:
    """연결 정보"""
    id: str
    source_id: str
    target_id: str
    connection_type: ConnectionType
    strength: float
    confidence: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_current_strength(self) -> float:
        """현재 강도 계산 (시간 감쇠 적용)"""
        days_since_access = (datetime.now() - self.last_accessed).days
        decay_factor = math.exp(-self.decay_rate * days_since_access)
        return self.strength * decay_factor

    def get_strength_category(self) -> ConnectionStrength:
        """강도 카테고리 반환"""
        current_strength = self.calculate_current_strength()

        if current_strength < 0.2:
            return ConnectionStrength.VERY_WEAK
        elif current_strength < 0.4:
            return ConnectionStrength.WEAK
        elif current_strength < 0.6:
            return ConnectionStrength.MEDIUM
        elif current_strength < 0.8:
            return ConnectionStrength.STRONG
        else:
            return ConnectionStrength.VERY_STRONG

    def should_be_pruned(self, threshold: float) -> bool:
        """정리 대상 여부 판단"""
        current_strength = self.calculate_current_strength()

        # 강도가 임계값 미만
        if current_strength < threshold:
            return True

        # 오랫동안 접근되지 않음
        days_since_access = (datetime.now() - self.last_accessed).days
        if days_since_access > 90 and current_strength < 0.5:  # 90일 + 중간 강도 미만
            return True

        # 접근 빈도가 매우 낮음
        if self.access_count <= 1 and current_strength < 0.3:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'connection_type': self.connection_type.value,
            'strength': self.strength,
            'current_strength': self.calculate_current_strength(),
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'decay_rate': self.decay_rate,
            'strength_category': self.get_strength_category().value,
            'metadata': self.metadata
        }


@dataclass
class PruningResult:
    """정리 결과"""
    pruned_count: int
    preserved_count: int
    strengthened_count: int
    total_processed: int
    network_efficiency_improvement: float
    memory_freed: int  # bytes
    quality_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pruned_count': self.pruned_count,
            'preserved_count': self.preserved_count,
            'strengthened_count': self.strengthened_count,
            'total_processed': self.total_processed,
            'network_efficiency_improvement': self.network_efficiency_improvement,
            'memory_freed': self.memory_freed,
            'quality_metrics': self.quality_metrics
        }


class WeakConnectionPruner:
    """약한 연결 정리기 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 설정
        self.default_pruning_threshold = 0.3
        self.strengthening_threshold = 0.7
        self.max_connections_per_node = 50
        self.preserve_recent_connections_days = 7

        # 연결 저장소 (시뮬레이션)
        self.connections: Dict[str, Connection] = {}
        self.pruned_connections: Dict[str, Connection] = {}

        # 통계
        self.pruning_stats = {
            'total_prunings': 0,
            'connections_pruned': 0,
            'connections_strengthened': 0,
            'network_efficiency': 0.0
        }

    async def prune_connections(self, weak_connections: int, threshold: float) -> Dict[str, Any]:
        """약한 연결 정리 실행"""
        self.logger.info(f"약한 연결 정리 시작: {weak_connections}개 연결 검토")

        try:
            # 시뮬레이션된 연결 생성
            connections = self._generate_simulation_connections(weak_connections)

            # 연결 정리 처리
            result = await self._process_connection_pruning(connections, threshold)

            # 통계 업데이트
            self._update_pruning_stats(result)

            self.logger.info(f"약한 연결 정리 완료: {result.pruned_count}개 제거됨")
            return result.to_dict()

        except Exception as e:
            self.logger.error(f"약한 연결 정리 오류: {str(e)}")
            raise RuntimeError(f"약한 연결 정리 실패: {str(e)}")

    def _generate_simulation_connections(self, count: int) -> List[Connection]:
        """시뮬레이션 연결 생성"""
        connections = []

        connection_types = list(ConnectionType)

        for i in range(count):
            # 연결 강도 분포 (약한 연결이 더 많음)
            if i < count * 0.4:  # 40%는 매우 약함
                strength = 0.1 + (i / (count * 0.4)) * 0.1  # 0.1-0.2
            elif i < count * 0.7:  # 30%는 약함
                strength = 0.2 + ((i - count * 0.4) / (count * 0.3)) * 0.2  # 0.2-0.4
            elif i < count * 0.9:  # 20%는 중간
                strength = 0.4 + ((i - count * 0.7) / (count * 0.2)) * 0.2  # 0.4-0.6
            else:  # 10%는 강함
                strength = 0.6 + ((i - count * 0.9) / (count * 0.1)) * 0.4  # 0.6-1.0

            connection_type = connection_types[i % len(connection_types)]
            created_time = datetime.now() - timedelta(days=i % 365)
            last_access_time = created_time + timedelta(days=(i % 100))

            connection = Connection(
                id=f"conn_{i:06d}",
                source_id=f"node_{i % 1000:06d}",
                target_id=f"node_{(i + 1) % 1000:06d}",
                connection_type=connection_type,
                strength=strength,
                confidence=min(1.0, strength + 0.1),
                created_at=created_time,
                last_accessed=last_access_time,
                access_count=max(1, int(strength * 50)),
                decay_rate=0.02 + (0.08 * (1 - strength)),  # 약한 연결일수록 빠른 감쇠
                metadata={
                    'description': f'{connection_type.value} 연결 #{i}',
                    'context': 'simulation'
                }
            )

            connections.append(connection)

        return connections

    async def _process_connection_pruning(self, connections: List[Connection],
                                        threshold: float) -> PruningResult:
        """연결 정리 처리"""
        pruned_count = 0
        preserved_count = 0
        strengthened_count = 0
        memory_freed = 0

        # 연결을 노드별로 그룹화
        node_connections = self._group_connections_by_node(connections)

        for connection in connections:
            # 최근 생성된 연결은 보존
            days_since_creation = (datetime.now() - connection.created_at).days
            if days_since_creation <= self.preserve_recent_connections_days:
                preserved_count += 1
                self.connections[connection.id] = connection
                continue

            # 정리 대상 여부 판단
            if connection.should_be_pruned(threshold):
                # 연결 정리
                await self._prune_connection(connection)
                pruned_count += 1
                memory_freed += self._estimate_connection_memory_size(connection)

            elif connection.calculate_current_strength() > self.strengthening_threshold:
                # 강한 연결 강화
                await self._strengthen_connection(connection)
                strengthened_count += 1
                preserved_count += 1
                self.connections[connection.id] = connection

            else:
                # 일반 보존
                preserved_count += 1
                self.connections[connection.id] = connection

        # 노드별 연결 수 제한 적용
        additional_pruned = await self._apply_connection_limits(node_connections)
        pruned_count += additional_pruned

        # 네트워크 효율성 개선 계산
        efficiency_improvement = self._calculate_efficiency_improvement(
            len(connections), pruned_count, strengthened_count
        )

        # 품질 메트릭 계산
        quality_metrics = {
            'pruning_ratio': pruned_count / max(len(connections), 1),
            'preservation_ratio': preserved_count / max(len(connections), 1),
            'strengthening_ratio': strengthened_count / max(len(connections), 1),
            'network_density_improvement': efficiency_improvement
        }

        return PruningResult(
            pruned_count=pruned_count,
            preserved_count=preserved_count,
            strengthened_count=strengthened_count,
            total_processed=len(connections),
            network_efficiency_improvement=efficiency_improvement,
            memory_freed=memory_freed,
            quality_metrics=quality_metrics
        )

    def _group_connections_by_node(self, connections: List[Connection]) -> Dict[str, List[Connection]]:
        """노드별 연결 그룹화"""
        node_connections = {}

        for connection in connections:
            # 출발 노드
            if connection.source_id not in node_connections:
                node_connections[connection.source_id] = []
            node_connections[connection.source_id].append(connection)

            # 도착 노드
            if connection.target_id not in node_connections:
                node_connections[connection.target_id] = []
            node_connections[connection.target_id].append(connection)

        return node_connections

    async def _prune_connection(self, connection: Connection):
        """연결 정리"""
        # 정리된 연결을 별도 저장소로 이동
        self.pruned_connections[connection.id] = connection

        # 시뮬레이션 지연
        await asyncio.sleep(0.001)

    async def _strengthen_connection(self, connection: Connection):
        """연결 강화"""
        # 강도 증가
        connection.strength = min(1.0, connection.strength + 0.1)

        # 접근 시간 업데이트
        connection.last_accessed = datetime.now()
        connection.access_count += 1

        # 시뮬레이션 지연
        await asyncio.sleep(0.001)

    def _estimate_connection_memory_size(self, connection: Connection) -> int:
        """연결 메모리 크기 추정"""
        # 시뮬레이션된 메모리 크기 (바이트)
        base_size = 256  # 기본 연결 정보
        metadata_size = len(str(connection.metadata)) * 2  # 메타데이터
        return base_size + metadata_size

    async def _apply_connection_limits(self, node_connections: Dict[str, List[Connection]]) -> int:
        """노드별 연결 수 제한 적용"""
        additional_pruned = 0

        for node_id, connections in node_connections.items():
            if len(connections) > self.max_connections_per_node:
                # 가장 약한 연결부터 제거
                connections.sort(key=lambda c: c.calculate_current_strength())

                excess_count = len(connections) - self.max_connections_per_node
                for i in range(excess_count):
                    connection = connections[i]
                    await self._prune_connection(connection)
                    additional_pruned += 1

        return additional_pruned

    def _calculate_efficiency_improvement(self, total: int, pruned: int, strengthened: int) -> float:
        """네트워크 효율성 개선 계산"""
        if total == 0:
            return 0.0

        # 정리로 인한 효율성 개선
        pruning_improvement = (pruned / total) * 0.3

        # 강화로 인한 효율성 개선
        strengthening_improvement = (strengthened / total) * 0.5

        # 전체 효율성 개선
        total_improvement = pruning_improvement + strengthening_improvement

        return min(1.0, total_improvement)

    def _update_pruning_stats(self, result: PruningResult):
        """정리 통계 업데이트"""
        self.pruning_stats['total_prunings'] += 1
        self.pruning_stats['connections_pruned'] += result.pruned_count
        self.pruning_stats['connections_strengthened'] += result.strengthened_count
        self.pruning_stats['network_efficiency'] = (
            self.pruning_stats['network_efficiency'] * 0.9 +
            result.network_efficiency_improvement * 0.1
        )

    def get_pruning_statistics(self) -> Dict[str, Any]:
        """정리 통계 조회"""
        return {
            'active_connections': len(self.connections),
            'pruned_connections': len(self.pruned_connections),
            'connection_types': self._get_connection_type_distribution(),
            'strength_distribution': self._get_strength_distribution(),
            'pruning_stats': self.pruning_stats.copy()
        }

    def _get_connection_type_distribution(self) -> Dict[str, int]:
        """연결 유형 분포 조회"""
        distribution = {}

        for connection in self.connections.values():
            conn_type = connection.connection_type.value
            if conn_type not in distribution:
                distribution[conn_type] = 0
            distribution[conn_type] += 1

        return distribution

    def _get_strength_distribution(self) -> Dict[str, int]:
        """강도 분포 조회"""
        distribution = {}

        for connection in self.connections.values():
            strength_category = connection.get_strength_category().value
            if strength_category not in distribution:
                distribution[strength_category] = 0
            distribution[strength_category] += 1

        return distribution

    async def analyze_network_health(self) -> Dict[str, Any]:
        """네트워크 건강도 분석"""
        self.logger.info("네트워크 건강도 분석 시작")

        if not self.connections:
            return {'error': '분석할 연결이 없습니다.'}

        # 건강도 분석 시뮬레이션
        await asyncio.sleep(0.3)

        # 전체 강도 평균
        total_strength = sum(c.calculate_current_strength() for c in self.connections.values())
        average_strength = total_strength / len(self.connections)

        # 네트워크 밀도
        total_possible_connections = len(self.connections) * (len(self.connections) - 1) / 2
        network_density = len(self.connections) / total_possible_connections if total_possible_connections > 0 else 0

        # 약한 연결 비율
        weak_connections = sum(1 for c in self.connections.values() if c.get_strength_category() in [ConnectionStrength.VERY_WEAK, ConnectionStrength.WEAK])
        weak_ratio = weak_connections / len(self.connections)

        health_metrics = {
            'average_strength': average_strength,
            'network_density': network_density,
            'weak_connection_ratio': weak_ratio,
            'total_connections': len(self.connections),
            'health_score': (average_strength * 0.4 + (1 - weak_ratio) * 0.4 + network_density * 0.2),
            'recommendations': self._generate_health_recommendations(average_strength, weak_ratio, network_density)
        }

        self.logger.info("네트워크 건강도 분석 완료")
        return health_metrics

    def _generate_health_recommendations(self, avg_strength: float, weak_ratio: float, density: float) -> List[str]:
        """건강도 기반 권장사항 생성"""
        recommendations = []

        if avg_strength < 0.5:
            recommendations.append("평균 연결 강도가 낮습니다. 연결 강화 프로세스를 개선하세요.")

        if weak_ratio > 0.6:
            recommendations.append("약한 연결의 비율이 높습니다. 정리 임계값을 높여보세요.")

        if density < 0.1:
            recommendations.append("네트워크 밀도가 낮습니다. 새로운 연결 생성을 고려하세요.")

        if density > 0.8:
            recommendations.append("네트워크가 과도하게 밀집되어 있습니다. 불필요한 연결을 정리하세요.")

        return recommendations

    async def restore_pruned_connections(self, connection_ids: List[str]) -> Dict[str, Any]:
        """정리된 연결 복원"""
        restored_count = 0
        not_found_count = 0

        for conn_id in connection_ids:
            if conn_id in self.pruned_connections:
                # 연결 복원
                connection = self.pruned_connections[conn_id]
                self.connections[conn_id] = connection
                del self.pruned_connections[conn_id]
                restored_count += 1
            else:
                not_found_count += 1

        return {
            'restored_count': restored_count,
            'not_found_count': not_found_count,
            'total_requested': len(connection_ids)
        }