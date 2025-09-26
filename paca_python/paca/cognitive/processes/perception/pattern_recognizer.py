"""
Pattern Recognizer

패턴 인식 시스템으로, 감각 데이터에서 의미 있는 패턴을 인식하고
분류합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID
import re
import hashlib

from ...base import BaseCognitiveProcessor


class PatternType(Enum):
    """패턴 유형"""
    SEQUENTIAL = auto()     # 순차적 패턴
    SPATIAL = auto()        # 공간적 패턴
    TEMPORAL = auto()       # 시간적 패턴
    STRUCTURAL = auto()     # 구조적 패턴
    SEMANTIC = auto()       # 의미적 패턴
    BEHAVIORAL = auto()     # 행동적 패턴


@dataclass
class Pattern:
    """패턴 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    type: PatternType = PatternType.STRUCTURAL
    signature: str = ""                     # 패턴 서명
    template: Any = None                    # 패턴 템플릿
    features: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.7       # 인식 임계값
    occurrence_count: int = 0               # 발생 횟수
    success_rate: float = 1.0              # 성공률
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


@dataclass
class PatternMatchResult:
    """패턴 매치 결과"""
    pattern_id: UUID
    pattern_name: str
    match_confidence: float = 0.0
    match_location: Optional[Dict[str, Any]] = None
    matched_features: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionConfig:
    """인식 설정"""
    max_patterns: int = 1000               # 최대 패턴 수
    similarity_threshold: float = 0.8      # 유사도 임계값
    learning_rate: float = 0.1             # 학습률
    pattern_decay_rate: float = 0.01       # 패턴 감쇠율
    enable_online_learning: bool = True     # 온라인 학습 활성화
    max_match_candidates: int = 10         # 최대 매치 후보 수


class PatternRecognizer(BaseCognitiveProcessor):
    """
    패턴 인식 시스템

    다양한 유형의 패턴을 학습하고 인식하여,
    감각 데이터에서 의미 있는 구조를 발견합니다.
    """

    def __init__(self, config: Optional[RecognitionConfig] = None):
        super().__init__()
        self.config = config or RecognitionConfig()

        # 패턴 저장소
        self._patterns: Dict[UUID, Pattern] = {}
        self._pattern_index: Dict[str, Set[UUID]] = {}  # 빠른 검색용 인덱스

        # 학습 상태
        self._learning_enabled = self.config.enable_online_learning
        self._recognition_history: List[PatternMatchResult] = []

        # 성능 메트릭
        self._total_recognitions = 0
        self._successful_recognitions = 0
        self._false_positives = 0

        # 특화된 인식기들
        self._text_recognizer = TextPatternRecognizer()
        self._sequence_recognizer = SequencePatternRecognizer()
        self._spatial_recognizer = SpatialPatternRecognizer()

    async def initialize(self) -> bool:
        """패턴 인식기 초기화"""
        try:
            self.logger.info("Initializing Pattern Recognizer...")

            # 기본 패턴 로드
            await self._load_default_patterns()

            # 서브 인식기 초기화
            await self._text_recognizer.initialize()
            await self._sequence_recognizer.initialize()
            await self._spatial_recognizer.initialize()

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._pattern_maintenance())

            self.logger.info("Pattern Recognizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Pattern Recognizer: {e}")
            return False

    async def recognize(self, data: Any, modality: str = "general") -> List[Dict[str, Any]]:
        """
        패턴 인식 수행

        Args:
            data: 인식할 데이터
            modality: 감각 양상

        Returns:
            인식된 패턴들의 목록
        """
        try:
            start_time = time.time()
            recognized_patterns = []

            # 데이터 전처리
            preprocessed_data = await self._preprocess_data(data, modality)

            # 적절한 인식기 선택 및 실행
            if modality == "textual" or isinstance(data, str):
                text_patterns = await self._text_recognizer.recognize(preprocessed_data)
                recognized_patterns.extend(text_patterns)

            elif modality == "sequential" or isinstance(data, (list, tuple)):
                seq_patterns = await self._sequence_recognizer.recognize(preprocessed_data)
                recognized_patterns.extend(seq_patterns)

            elif modality == "spatial":
                spatial_patterns = await self._spatial_recognizer.recognize(preprocessed_data)
                recognized_patterns.extend(spatial_patterns)

            # 일반 패턴 매칭
            general_patterns = await self._general_pattern_matching(preprocessed_data)
            recognized_patterns.extend(general_patterns)

            # 결과 후처리
            processed_results = await self._postprocess_results(recognized_patterns)

            # 학습 및 적응
            if self._learning_enabled:
                await self._learn_from_recognition(data, processed_results)

            # 메트릭 업데이트
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processed_results, processing_time)

            return processed_results

        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {e}")
            return []

    async def recognize_parallel(self, data: Any, modality: str = "general") -> Dict[str, Any]:
        """병렬 패턴 인식"""
        try:
            # 여러 인식기를 병렬로 실행
            tasks = []

            if isinstance(data, str):
                tasks.append(self._text_recognizer.recognize(data))

            if isinstance(data, (list, tuple)):
                tasks.append(self._sequence_recognizer.recognize(data))

            tasks.append(self._general_pattern_matching(data))

            # 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 통합
            all_patterns = []
            for result in results:
                if isinstance(result, list):
                    all_patterns.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"Parallel recognition task failed: {result}")

            return {"patterns": all_patterns}

        except Exception as e:
            self.logger.error(f"Error in parallel recognition: {e}")
            return {"patterns": []}

    async def learn_pattern(self, data: Any, pattern_name: str,
                          pattern_type: PatternType = PatternType.STRUCTURAL) -> bool:
        """
        새로운 패턴 학습

        Args:
            data: 패턴 데이터
            pattern_name: 패턴 이름
            pattern_type: 패턴 유형

        Returns:
            학습 성공 여부
        """
        try:
            # 패턴 서명 생성
            signature = await self._generate_pattern_signature(data, pattern_type)

            # 중복 패턴 확인
            existing_pattern = await self._find_similar_pattern(signature)
            if existing_pattern:
                # 기존 패턴 업데이트
                existing_pattern.occurrence_count += 1
                existing_pattern.last_seen = time.time()
                return True

            # 새 패턴 생성
            pattern = Pattern(
                name=pattern_name,
                type=pattern_type,
                signature=signature,
                template=data,
                features=await self._extract_pattern_features(data, pattern_type)
            )

            # 패턴 저장
            self._patterns[pattern.id] = pattern
            await self._update_pattern_index(pattern)

            self.logger.info(f"Learned new pattern: {pattern_name} ({pattern_type.name})")
            return True

        except Exception as e:
            self.logger.error(f"Error learning pattern {pattern_name}: {e}")
            return False

    async def verify_pattern(self, data: Any, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 검증"""
        try:
            pattern_id = pattern.get("id")
            if not pattern_id:
                return {"verified": False, "reason": "Invalid pattern ID"}

            # UUID 변환
            if isinstance(pattern_id, str):
                try:
                    pattern_id = UUID(pattern_id)
                except ValueError:
                    return {"verified": False, "reason": "Invalid UUID format"}

            if pattern_id not in self._patterns:
                return {"verified": False, "reason": "Pattern not found"}

            stored_pattern = self._patterns[pattern_id]

            # 패턴 매칭 수행
            match_result = await self._match_pattern(data, stored_pattern)

            verified = match_result.match_confidence >= stored_pattern.confidence_threshold

            return {
                "verified": verified,
                "confidence": match_result.match_confidence,
                "similarity": match_result.similarity_score,
                "pattern_name": stored_pattern.name
            }

        except Exception as e:
            self.logger.error(f"Error verifying pattern: {e}")
            return {"verified": False, "reason": str(e)}

    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """패턴 통계 조회"""
        pattern_counts = {}
        for pattern in self._patterns.values():
            pattern_type = pattern.type.name
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

        return {
            "total_patterns": len(self._patterns),
            "pattern_types": pattern_counts,
            "total_recognitions": self._total_recognitions,
            "success_rate": (
                self._successful_recognitions / max(1, self._total_recognitions)
            ),
            "false_positive_rate": (
                self._false_positives / max(1, self._total_recognitions)
            ),
            "learning_enabled": self._learning_enabled
        }

    async def _preprocess_data(self, data: Any, modality: str) -> Any:
        """데이터 전처리"""
        try:
            if modality == "textual" and isinstance(data, str):
                # 텍스트 정규화
                return re.sub(r'\s+', ' ', data.strip().lower())

            elif modality == "sequential" and isinstance(data, (list, tuple)):
                # 시퀀스 정규화
                return list(data)

            return data

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return data

    async def _general_pattern_matching(self, data: Any) -> List[Dict[str, Any]]:
        """일반 패턴 매칭"""
        try:
            matches = []

            # 모든 패턴에 대해 매칭 시도
            for pattern in self._patterns.values():
                match_result = await self._match_pattern(data, pattern)

                if match_result.match_confidence >= pattern.confidence_threshold:
                    matches.append({
                        "id": str(pattern.id),
                        "name": pattern.name,
                        "type": pattern.type.name,
                        "confidence": match_result.match_confidence,
                        "similarity": match_result.similarity_score,
                        "features": match_result.matched_features,
                        "location": match_result.match_location
                    })

            # 신뢰도 순으로 정렬
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            return matches[:self.config.max_match_candidates]

        except Exception as e:
            self.logger.error(f"Error in general pattern matching: {e}")
            return []

    async def _match_pattern(self, data: Any, pattern: Pattern) -> PatternMatchResult:
        """개별 패턴 매칭"""
        try:
            start_time = time.time()

            # 패턴 유형별 매칭 전략
            if pattern.type == PatternType.SEQUENTIAL:
                confidence, similarity = await self._match_sequential_pattern(data, pattern)
            elif pattern.type == PatternType.SPATIAL:
                confidence, similarity = await self._match_spatial_pattern(data, pattern)
            elif pattern.type == PatternType.SEMANTIC:
                confidence, similarity = await self._match_semantic_pattern(data, pattern)
            else:
                confidence, similarity = await self._match_structural_pattern(data, pattern)

            processing_time = (time.time() - start_time) * 1000

            return PatternMatchResult(
                pattern_id=pattern.id,
                pattern_name=pattern.name,
                match_confidence=confidence,
                similarity_score=similarity,
                processing_time_ms=processing_time
            )

        except Exception as e:
            self.logger.error(f"Error matching pattern {pattern.name}: {e}")
            return PatternMatchResult(
                pattern_id=pattern.id,
                pattern_name=pattern.name,
                match_confidence=0.0
            )

    async def _match_structural_pattern(self, data: Any, pattern: Pattern) -> tuple[float, float]:
        """구조적 패턴 매칭"""
        try:
            if pattern.template is None:
                return 0.0, 0.0

            # 데이터 구조 비교
            if type(data) != type(pattern.template):
                return 0.0, 0.0

            if isinstance(data, str):
                # 문자열 유사도 계산
                similarity = await self._calculate_string_similarity(data, pattern.template)
                confidence = similarity if similarity > 0.5 else 0.0

            elif isinstance(data, (list, tuple)):
                # 시퀀스 유사도 계산
                similarity = await self._calculate_sequence_similarity(data, pattern.template)
                confidence = similarity if similarity > 0.6 else 0.0

            elif isinstance(data, dict):
                # 딕셔너리 구조 유사도
                similarity = await self._calculate_dict_similarity(data, pattern.template)
                confidence = similarity if similarity > 0.7 else 0.0

            else:
                # 기본 동등성 비교
                similarity = 1.0 if data == pattern.template else 0.0
                confidence = similarity

            return confidence, similarity

        except Exception as e:
            self.logger.error(f"Error in structural pattern matching: {e}")
            return 0.0, 0.0

    async def _match_sequential_pattern(self, data: Any, pattern: Pattern) -> tuple[float, float]:
        """순차적 패턴 매칭"""
        if not isinstance(data, (list, tuple)) or not isinstance(pattern.template, (list, tuple)):
            return 0.0, 0.0

        return await self._calculate_sequence_similarity(data, pattern.template), 0.8

    async def _match_spatial_pattern(self, data: Any, pattern: Pattern) -> tuple[float, float]:
        """공간적 패턴 매칭"""
        # 간단한 구현: 공간 좌표나 구조가 있는 경우
        if isinstance(data, dict) and isinstance(pattern.template, dict):
            spatial_keys = {"x", "y", "z", "position", "location"}
            data_spatial = {k: v for k, v in data.items() if k in spatial_keys}
            pattern_spatial = {k: v for k, v in pattern.template.items() if k in spatial_keys}

            if data_spatial and pattern_spatial:
                similarity = await self._calculate_dict_similarity(data_spatial, pattern_spatial)
                return similarity, similarity

        return 0.0, 0.0

    async def _match_semantic_pattern(self, data: Any, pattern: Pattern) -> tuple[float, float]:
        """의미적 패턴 매칭"""
        if isinstance(data, str) and isinstance(pattern.template, str):
            # 키워드 기반 의미 유사도
            data_words = set(data.lower().split())
            pattern_words = set(pattern.template.lower().split())

            if not pattern_words:
                return 0.0, 0.0

            intersection = data_words & pattern_words
            similarity = len(intersection) / len(pattern_words)
            confidence = similarity if similarity > 0.3 else 0.0

            return confidence, similarity

        return 0.0, 0.0

    async def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """문자열 유사도 계산 (Levenshtein 거리 기반)"""
        try:
            if not str1 or not str2:
                return 0.0

            # 간단한 Levenshtein 거리 계산
            len1, len2 = len(str1), len(str2)
            dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

            for i in range(len1 + 1):
                dp[i][0] = i
            for j in range(len2 + 1):
                dp[0][j] = j

            for i in range(1, len1 + 1):
                for j in range(1, len2 + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

            max_len = max(len1, len2)
            if max_len == 0:
                return 1.0

            similarity = 1.0 - dp[len1][len2] / max_len
            return max(0.0, similarity)

        except Exception as e:
            self.logger.error(f"Error calculating string similarity: {e}")
            return 0.0

    async def _calculate_sequence_similarity(self, seq1: Union[list, tuple],
                                           seq2: Union[list, tuple]) -> float:
        """시퀀스 유사도 계산"""
        try:
            if not seq1 or not seq2:
                return 0.0

            # 공통 요소 비율 계산
            set1, set2 = set(seq1), set(seq2)
            intersection = set1 & set2
            union = set1 | set2

            jaccard_similarity = len(intersection) / len(union) if union else 0.0

            # 순서 고려한 유사도
            order_penalty = 0.0
            min_len = min(len(seq1), len(seq2))

            for i in range(min_len):
                if seq1[i] != seq2[i]:
                    order_penalty += 1.0 / min_len

            order_similarity = 1.0 - order_penalty

            # 가중 평균
            return (jaccard_similarity * 0.7 + order_similarity * 0.3)

        except Exception as e:
            self.logger.error(f"Error calculating sequence similarity: {e}")
            return 0.0

    async def _calculate_dict_similarity(self, dict1: dict, dict2: dict) -> float:
        """딕셔너리 유사도 계산"""
        try:
            if not dict1 or not dict2:
                return 0.0

            common_keys = set(dict1.keys()) & set(dict2.keys())
            all_keys = set(dict1.keys()) | set(dict2.keys())

            if not all_keys:
                return 1.0

            # 키 일치도
            key_similarity = len(common_keys) / len(all_keys)

            # 값 일치도
            value_similarity = 0.0
            if common_keys:
                matching_values = sum(
                    1 for key in common_keys if dict1[key] == dict2[key]
                )
                value_similarity = matching_values / len(common_keys)

            return (key_similarity * 0.5 + value_similarity * 0.5)

        except Exception as e:
            self.logger.error(f"Error calculating dict similarity: {e}")
            return 0.0

    async def _generate_pattern_signature(self, data: Any, pattern_type: PatternType) -> str:
        """패턴 서명 생성"""
        try:
            # 데이터를 문자열로 변환
            if isinstance(data, str):
                content = data
            elif isinstance(data, (list, tuple)):
                content = str(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else data)
            elif isinstance(data, dict):
                content = str(sorted(data.items()))
            else:
                content = str(data)

            # 패턴 유형과 함께 해시 생성
            signature_input = f"{pattern_type.name}_{content}"
            signature = hashlib.md5(signature_input.encode()).hexdigest()

            return signature

        except Exception as e:
            self.logger.error(f"Error generating pattern signature: {e}")
            return str(uuid4())

    async def _extract_pattern_features(self, data: Any, pattern_type: PatternType) -> Dict[str, Any]:
        """패턴 특징 추출"""
        features = {}

        try:
            if isinstance(data, str):
                features.update({
                    "length": len(data),
                    "word_count": len(data.split()),
                    "char_types": len(set(data)),
                    "has_numbers": any(c.isdigit() for c in data),
                    "has_uppercase": any(c.isupper() for c in data)
                })

            elif isinstance(data, (list, tuple)):
                features.update({
                    "length": len(data),
                    "unique_elements": len(set(data)),
                    "element_types": list(set(type(x).__name__ for x in data))
                })

            elif isinstance(data, dict):
                features.update({
                    "key_count": len(data),
                    "value_types": list(set(type(v).__name__ for v in data.values())),
                    "nested_depth": self._calculate_dict_depth(data)
                })

            features["pattern_type"] = pattern_type.name
            features["data_type"] = type(data).__name__

        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {e}")

        return features

    def _calculate_dict_depth(self, d: dict, current_depth: int = 1) -> int:
        """딕셔너리 중첩 깊이 계산"""
        if not isinstance(d, dict):
            return current_depth

        max_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    async def _find_similar_pattern(self, signature: str) -> Optional[Pattern]:
        """유사한 패턴 찾기"""
        for pattern in self._patterns.values():
            if pattern.signature == signature:
                return pattern
        return None

    async def _update_pattern_index(self, pattern: Pattern) -> None:
        """패턴 인덱스 업데이트"""
        try:
            # 패턴 유형별 인덱스
            type_key = pattern.type.name
            if type_key not in self._pattern_index:
                self._pattern_index[type_key] = set()
            self._pattern_index[type_key].add(pattern.id)

            # 이름 기반 인덱스
            name_key = pattern.name.lower()
            if name_key not in self._pattern_index:
                self._pattern_index[name_key] = set()
            self._pattern_index[name_key].add(pattern.id)

        except Exception as e:
            self.logger.error(f"Error updating pattern index: {e}")

    async def _postprocess_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 후처리"""
        try:
            # 중복 제거
            seen_patterns = set()
            filtered_results = []

            for result in results:
                pattern_key = (result.get("name", ""), result.get("confidence", 0))
                if pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    filtered_results.append(result)

            # 신뢰도 기준 필터링
            threshold = self.config.similarity_threshold
            high_confidence_results = [
                r for r in filtered_results
                if r.get("confidence", 0) >= threshold
            ]

            return high_confidence_results

        except Exception as e:
            self.logger.error(f"Error postprocessing results: {e}")
            return results

    async def _learn_from_recognition(self, data: Any, results: List[Dict[str, Any]]) -> None:
        """인식 결과로부터 학습"""
        if not self._learning_enabled:
            return

        try:
            # 성공적인 인식에 대한 패턴 강화
            for result in results:
                pattern_name = result.get("name")
                confidence = result.get("confidence", 0)

                # 해당 패턴 찾기
                pattern = None
                for p in self._patterns.values():
                    if p.name == pattern_name:
                        pattern = p
                        break

                if pattern:
                    # 패턴 통계 업데이트
                    pattern.occurrence_count += 1
                    pattern.last_seen = time.time()

                    # 성공률 업데이트
                    if confidence >= pattern.confidence_threshold:
                        pattern.success_rate = (
                            pattern.success_rate * 0.9 + 1.0 * 0.1
                        )
                    else:
                        pattern.success_rate = (
                            pattern.success_rate * 0.9 + 0.0 * 0.1
                        )

        except Exception as e:
            self.logger.error(f"Error learning from recognition: {e}")

    def _update_metrics(self, results: List[Dict[str, Any]], processing_time: float) -> None:
        """메트릭 업데이트"""
        self._total_recognitions += 1

        if results:
            self._successful_recognitions += 1

            # 높은 신뢰도 결과 개수 확인
            high_confidence_count = sum(
                1 for r in results if r.get("confidence", 0) >= 0.9
            )

            if high_confidence_count == 0:
                self._false_positives += 1

    async def _load_default_patterns(self) -> None:
        """기본 패턴 로드"""
        try:
            # 기본 텍스트 패턴들
            default_patterns = [
                {
                    "name": "email_pattern",
                    "type": PatternType.STRUCTURAL,
                    "template": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                },
                {
                    "name": "url_pattern",
                    "type": PatternType.STRUCTURAL,
                    "template": r"https?://[^\s]+"
                },
                {
                    "name": "number_sequence",
                    "type": PatternType.SEQUENTIAL,
                    "template": [1, 2, 3, 4, 5]
                },
                {
                    "name": "greeting_pattern",
                    "type": PatternType.SEMANTIC,
                    "template": "hello hi hey greetings"
                }
            ]

            for pattern_data in default_patterns:
                await self.learn_pattern(
                    pattern_data["template"],
                    pattern_data["name"],
                    pattern_data["type"]
                )

        except Exception as e:
            self.logger.error(f"Error loading default patterns: {e}")

    async def _pattern_maintenance(self) -> None:
        """패턴 유지보수 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()

                # 오래되거나 성능이 낮은 패턴 정리
                patterns_to_remove = []

                for pattern_id, pattern in self._patterns.items():
                    # 30일 이상 사용되지 않은 패턴
                    if current_time - pattern.last_seen > 30 * 24 * 3600:
                        patterns_to_remove.append(pattern_id)
                    # 성공률이 매우 낮은 패턴
                    elif pattern.success_rate < 0.1 and pattern.occurrence_count > 10:
                        patterns_to_remove.append(pattern_id)

                # 패턴 제거
                for pattern_id in patterns_to_remove:
                    if pattern_id in self._patterns:
                        removed_pattern = self._patterns[pattern_id]
                        del self._patterns[pattern_id]
                        self.logger.info(f"Removed low-performance pattern: {removed_pattern.name}")

                # 패턴 수 제한
                if len(self._patterns) > self.config.max_patterns:
                    # 성능이 낮은 패턴부터 제거
                    sorted_patterns = sorted(
                        self._patterns.values(),
                        key=lambda p: p.success_rate * p.occurrence_count
                    )

                    remove_count = len(self._patterns) - self.config.max_patterns
                    for i in range(remove_count):
                        pattern_to_remove = sorted_patterns[i]
                        del self._patterns[pattern_to_remove.id]

                await asyncio.sleep(3600)  # 1시간마다 실행

            except Exception as e:
                self.logger.error(f"Error in pattern maintenance: {e}")
                await asyncio.sleep(3600)


class TextPatternRecognizer:
    """텍스트 전용 패턴 인식기"""

    async def initialize(self) -> None:
        """초기화"""
        pass

    async def recognize(self, text: str) -> List[Dict[str, Any]]:
        """텍스트 패턴 인식"""
        patterns = []

        try:
            # 이메일 패턴
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if re.search(email_pattern, text):
                patterns.append({
                    "name": "email",
                    "type": "STRUCTURAL",
                    "confidence": 0.95,
                    "matches": re.findall(email_pattern, text)
                })

            # URL 패턴
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, text)
            if urls:
                patterns.append({
                    "name": "url",
                    "type": "STRUCTURAL",
                    "confidence": 0.9,
                    "matches": urls
                })

            # 숫자 패턴
            number_pattern = r'\d+'
            numbers = re.findall(number_pattern, text)
            if numbers:
                patterns.append({
                    "name": "numbers",
                    "type": "STRUCTURAL",
                    "confidence": 0.8,
                    "matches": numbers
                })

        except Exception as e:
            pass

        return patterns


class SequencePatternRecognizer:
    """시퀀스 전용 패턴 인식기"""

    async def initialize(self) -> None:
        """초기화"""
        pass

    async def recognize(self, sequence: Union[list, tuple]) -> List[Dict[str, Any]]:
        """시퀀스 패턴 인식"""
        patterns = []

        try:
            if len(sequence) < 2:
                return patterns

            # 증가 패턴
            if self._is_increasing_sequence(sequence):
                patterns.append({
                    "name": "increasing_sequence",
                    "type": "SEQUENTIAL",
                    "confidence": 0.9
                })

            # 감소 패턴
            if self._is_decreasing_sequence(sequence):
                patterns.append({
                    "name": "decreasing_sequence",
                    "type": "SEQUENTIAL",
                    "confidence": 0.9
                })

            # 반복 패턴
            repetition = self._find_repetition_pattern(sequence)
            if repetition:
                patterns.append({
                    "name": "repetition_pattern",
                    "type": "SEQUENTIAL",
                    "confidence": 0.85,
                    "pattern": repetition
                })

        except Exception as e:
            pass

        return patterns

    def _is_increasing_sequence(self, seq: Union[list, tuple]) -> bool:
        """증가 시퀀스 확인"""
        try:
            for i in range(1, len(seq)):
                if seq[i] <= seq[i-1]:
                    return False
            return True
        except:
            return False

    def _is_decreasing_sequence(self, seq: Union[list, tuple]) -> bool:
        """감소 시퀀스 확인"""
        try:
            for i in range(1, len(seq)):
                if seq[i] >= seq[i-1]:
                    return False
            return True
        except:
            return False

    def _find_repetition_pattern(self, seq: Union[list, tuple]) -> Optional[List]:
        """반복 패턴 찾기"""
        try:
            for pattern_length in range(1, len(seq) // 2 + 1):
                pattern = seq[:pattern_length]
                repeats = len(seq) // pattern_length

                reconstructed = pattern * repeats
                if reconstructed == seq[:len(reconstructed)]:
                    return pattern

            return None
        except:
            return None


class SpatialPatternRecognizer:
    """공간 패턴 인식기"""

    async def initialize(self) -> None:
        """초기화"""
        pass

    async def recognize(self, data: Any) -> List[Dict[str, Any]]:
        """공간 패턴 인식"""
        patterns = []

        try:
            if isinstance(data, dict):
                # 좌표 패턴 인식
                if self._has_coordinate_pattern(data):
                    patterns.append({
                        "name": "coordinate_pattern",
                        "type": "SPATIAL",
                        "confidence": 0.8
                    })

        except Exception as e:
            pass

        return patterns

    def _has_coordinate_pattern(self, data: dict) -> bool:
        """좌표 패턴 확인"""
        coordinate_keys = {"x", "y", "z", "lat", "lon", "latitude", "longitude"}
        return len(set(data.keys()) & coordinate_keys) >= 2


async def create_pattern_recognizer(config: Optional[RecognitionConfig] = None) -> PatternRecognizer:
    """PatternRecognizer 인스턴스 생성 및 초기화"""
    recognizer = PatternRecognizer(config)
    await recognizer.initialize()
    return recognizer