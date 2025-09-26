"""
Sensory Processor

감각 데이터 처리 시스템으로, 다양한 감각 양상의 원시 데이터를
처리하고 특징을 추출합니다.
"""

import asyncio
import time
import re
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class SensoryModality(Enum):
    """감각 양상"""
    VISUAL = "visual"           # 시각
    AUDITORY = "auditory"       # 청각
    TEXTUAL = "textual"         # 텍스트
    TEMPORAL = "temporal"       # 시간
    SPATIAL = "spatial"         # 공간
    SEMANTIC = "semantic"       # 의미
    NUMERICAL = "numerical"     # 수치
    STRUCTURAL = "structural"   # 구조


class ProcessingStage(Enum):
    """처리 단계"""
    RAW_INPUT = auto()          # 원시 입력
    PREPROCESSING = auto()      # 전처리
    FEATURE_EXTRACTION = auto() # 특징 추출
    NORMALIZATION = auto()      # 정규화
    ENHANCEMENT = auto()        # 향상
    VALIDATION = auto()         # 검증


@dataclass
class SensoryData:
    """감각 데이터"""
    id: UUID = field(default_factory=uuid4)
    modality: SensoryModality = SensoryModality.TEXTUAL
    raw_data: Any = None
    processed_data: Any = None
    features: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    confidence: float = 1.0
    processing_stage: ProcessingStage = ProcessingStage.RAW_INPUT
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SensoryResult:
    """감각 처리 결과"""
    input_id: UUID
    modality: SensoryModality
    processed_data: Any = None
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class SensoryProcessor(BaseCognitiveProcessor):
    """
    감각 데이터 처리 시스템

    다양한 감각 양상의 데이터를 전처리하고 특징을 추출하여
    상위 인지 프로세스에서 활용할 수 있도록 변환합니다.
    """

    def __init__(self):
        super().__init__()

        # 처리기 매핑
        self._processors: Dict[SensoryModality, Callable] = {}

        # 품질 관리
        self._quality_thresholds = {
            SensoryModality.TEXTUAL: 0.7,
            SensoryModality.VISUAL: 0.6,
            SensoryModality.AUDITORY: 0.6,
            SensoryModality.NUMERICAL: 0.8,
            SensoryModality.STRUCTURAL: 0.7
        }

        # 성능 메트릭
        self._total_processed = 0
        self._successful_processed = 0
        self._processing_times: List[float] = []

    async def initialize(self) -> bool:
        """감각 처리기 초기화"""
        try:
            self.logger.info("Initializing Sensory Processor...")

            # 양상별 처리기 등록
            await self._register_processors()

            self.logger.info("Sensory Processor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Sensory Processor: {e}")
            return False

    async def preprocess(self, data: Any, modality: str) -> Any:
        """
        데이터 전처리

        Args:
            data: 전처리할 원시 데이터
            modality: 감각 양상

        Returns:
            전처리된 데이터
        """
        try:
            # 양상 타입 변환
            sensory_modality = self._get_sensory_modality(modality)

            # 해당 양상의 전처리기 실행
            if sensory_modality in self._processors:
                processor = self._processors[sensory_modality]
                return await processor(data, "preprocess")

            # 기본 전처리
            return await self._default_preprocess(data, sensory_modality)

        except Exception as e:
            self.logger.error(f"Error preprocessing {modality} data: {e}")
            return data

    async def extract_features(self, data: Any, modality: str) -> Dict[str, Any]:
        """
        특징 추출

        Args:
            data: 특징을 추출할 데이터
            modality: 감각 양상

        Returns:
            추출된 특징들
        """
        try:
            # 양상 타입 변환
            sensory_modality = self._get_sensory_modality(modality)

            # 해당 양상의 특징 추출기 실행
            if sensory_modality in self._processors:
                processor = self._processors[sensory_modality]
                return await processor(data, "extract_features")

            # 기본 특징 추출
            return await self._default_feature_extraction(data, sensory_modality)

        except Exception as e:
            self.logger.error(f"Error extracting features from {modality} data: {e}")
            return {}

    async def process_parallel(self, data: Any, modality: str) -> Dict[str, Any]:
        """병렬 처리"""
        try:
            # 병렬로 전처리와 특징 추출 수행
            preprocess_task = asyncio.create_task(self.preprocess(data, modality))
            feature_task = asyncio.create_task(self.extract_features(data, modality))

            processed_data, features = await asyncio.gather(
                preprocess_task, feature_task
            )

            return {
                "objects": [{"data": processed_data, "confidence": 0.8}],
                "patterns": [{"features": features, "confidence": 0.7}]
            }

        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            return {"objects": [], "patterns": []}

    async def process_sensory_data(self, sensory_data: SensoryData) -> SensoryResult:
        """
        감각 데이터 종합 처리

        Args:
            sensory_data: 처리할 감각 데이터

        Returns:
            처리 결과
        """
        try:
            start_time = time.time()

            # 1. 전처리
            sensory_data.processing_stage = ProcessingStage.PREPROCESSING
            processed_data = await self.preprocess(sensory_data.raw_data, sensory_data.modality.value)

            # 2. 특징 추출
            sensory_data.processing_stage = ProcessingStage.FEATURE_EXTRACTION
            features = await self.extract_features(processed_data, sensory_data.modality.value)

            # 3. 품질 평가
            sensory_data.processing_stage = ProcessingStage.VALIDATION
            quality_metrics = await self._assess_quality(processed_data, features, sensory_data.modality)

            # 4. 정규화
            sensory_data.processing_stage = ProcessingStage.NORMALIZATION
            normalized_data = await self._normalize_data(processed_data, sensory_data.modality)

            # 처리 시간 계산
            processing_time = (time.time() - start_time) * 1000

            # 결과 생성
            result = SensoryResult(
                input_id=sensory_data.id,
                modality=sensory_data.modality,
                processed_data=normalized_data,
                extracted_features=features,
                quality_metrics=quality_metrics,
                processing_time_ms=processing_time,
                success=quality_metrics.get("overall_quality", 0) >= self._quality_thresholds.get(sensory_data.modality, 0.5)
            )

            # 메트릭 업데이트
            self._update_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing sensory data: {e}")
            return SensoryResult(
                input_id=sensory_data.id,
                modality=sensory_data.modality,
                success=False,
                error_message=str(e)
            )

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계 조회"""
        return {
            "total_processed": self._total_processed,
            "success_rate": (
                self._successful_processed / max(1, self._total_processed)
            ),
            "average_processing_time_ms": (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times else 0
            ),
            "supported_modalities": [mod.value for mod in self._processors.keys()],
            "quality_thresholds": {
                mod.value: threshold for mod, threshold in self._quality_thresholds.items()
            }
        }

    def _get_sensory_modality(self, modality: str) -> SensoryModality:
        """문자열을 SensoryModality로 변환"""
        try:
            # 직접 매핑
            for mod in SensoryModality:
                if mod.value == modality.lower():
                    return mod

            # 유사성 기반 매핑
            if "text" in modality.lower():
                return SensoryModality.TEXTUAL
            elif "visual" in modality.lower() or "image" in modality.lower():
                return SensoryModality.VISUAL
            elif "audio" in modality.lower() or "sound" in modality.lower():
                return SensoryModality.AUDITORY
            elif "number" in modality.lower() or "numeric" in modality.lower():
                return SensoryModality.NUMERICAL
            elif "time" in modality.lower() or "temporal" in modality.lower():
                return SensoryModality.TEMPORAL
            elif "space" in modality.lower() or "spatial" in modality.lower():
                return SensoryModality.SPATIAL
            elif "semantic" in modality.lower() or "meaning" in modality.lower():
                return SensoryModality.SEMANTIC
            else:
                return SensoryModality.STRUCTURAL

        except Exception as e:
            self.logger.error(f"Error converting modality {modality}: {e}")
            return SensoryModality.TEXTUAL

    async def _register_processors(self) -> None:
        """양상별 처리기 등록"""
        try:
            self._processors[SensoryModality.TEXTUAL] = self._process_textual
            self._processors[SensoryModality.NUMERICAL] = self._process_numerical
            self._processors[SensoryModality.TEMPORAL] = self._process_temporal
            self._processors[SensoryModality.SPATIAL] = self._process_spatial
            self._processors[SensoryModality.STRUCTURAL] = self._process_structural
            self._processors[SensoryModality.SEMANTIC] = self._process_semantic

        except Exception as e:
            self.logger.error(f"Error registering processors: {e}")

    async def _process_textual(self, data: Any, operation: str) -> Any:
        """텍스트 데이터 처리"""
        try:
            if not isinstance(data, str):
                data = str(data)

            if operation == "preprocess":
                # 텍스트 정리 및 정규화
                cleaned = re.sub(r'\s+', ' ', data.strip())
                cleaned = re.sub(r'[^\w\s\-.,!?;:]', '', cleaned)
                return cleaned.lower()

            elif operation == "extract_features":
                # 텍스트 특징 추출
                features = {
                    "length": len(data),
                    "word_count": len(data.split()),
                    "sentence_count": len([s for s in data.split('.') if s.strip()]),
                    "avg_word_length": (
                        sum(len(word) for word in data.split()) / max(1, len(data.split()))
                    ),
                    "has_numbers": bool(re.search(r'\d', data)),
                    "has_punctuation": bool(re.search(r'[.,!?;:]', data)),
                    "uppercase_ratio": (
                        sum(1 for c in data if c.isupper()) / max(1, len(data))
                    ),
                    "unique_words": len(set(data.lower().split())),
                    "complexity_score": self._calculate_text_complexity(data)
                }

                # 키워드 추출
                words = data.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # 3글자 이상 단어만
                        word_freq[word] = word_freq.get(word, 0) + 1

                # 상위 5개 키워드
                if word_freq:
                    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                    features["keywords"] = [word for word, freq in top_keywords]

                return features

        except Exception as e:
            self.logger.error(f"Error processing textual data: {e}")
            return data if operation == "preprocess" else {}

    async def _process_numerical(self, data: Any, operation: str) -> Any:
        """수치 데이터 처리"""
        try:
            if operation == "preprocess":
                # 수치 데이터 정규화
                if isinstance(data, (list, tuple)):
                    return [float(x) if isinstance(x, (int, float)) else 0.0 for x in data]
                elif isinstance(data, (int, float)):
                    return float(data)
                elif isinstance(data, str):
                    # 문자열에서 숫자 추출
                    numbers = re.findall(r'-?\d+\.?\d*', data)
                    return [float(n) for n in numbers]
                else:
                    return 0.0

            elif operation == "extract_features":
                # 수치 특징 추출
                if isinstance(data, (list, tuple)) and data:
                    numeric_data = [x for x in data if isinstance(x, (int, float))]
                    if numeric_data:
                        features = {
                            "count": len(numeric_data),
                            "sum": sum(numeric_data),
                            "mean": sum(numeric_data) / len(numeric_data),
                            "min": min(numeric_data),
                            "max": max(numeric_data),
                            "range": max(numeric_data) - min(numeric_data),
                            "variance": self._calculate_variance(numeric_data),
                            "is_sequence": self._is_numeric_sequence(numeric_data),
                            "trend": self._detect_trend(numeric_data)
                        }
                        return features

                elif isinstance(data, (int, float)):
                    return {
                        "value": data,
                        "is_integer": isinstance(data, int),
                        "is_positive": data > 0,
                        "magnitude": abs(data),
                        "digit_count": len(str(abs(int(data))))
                    }

                return {}

        except Exception as e:
            self.logger.error(f"Error processing numerical data: {e}")
            return data if operation == "preprocess" else {}

    async def _process_temporal(self, data: Any, operation: str) -> Any:
        """시간 데이터 처리"""
        try:
            if operation == "preprocess":
                # 시간 데이터 정규화
                if isinstance(data, (int, float)):
                    return data  # Unix timestamp로 가정
                elif isinstance(data, str):
                    # 간단한 시간 패턴 인식
                    time_patterns = [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
                        r'\d{1,2}/\d{1,2}/\d{4}'  # MM/DD/YYYY
                    ]
                    for pattern in time_patterns:
                        if re.search(pattern, data):
                            return data  # 시간 형식 발견
                    return time.time()  # 현재 시간으로 기본값
                else:
                    return time.time()

            elif operation == "extract_features":
                current_time = time.time()
                if isinstance(data, (int, float)):
                    timestamp = data
                else:
                    timestamp = current_time

                features = {
                    "timestamp": timestamp,
                    "is_past": timestamp < current_time,
                    "is_future": timestamp > current_time,
                    "age_seconds": abs(current_time - timestamp),
                    "hour_of_day": int((timestamp % 86400) / 3600),
                    "day_of_week": int((timestamp / 86400) % 7),
                    "is_recent": abs(current_time - timestamp) < 3600  # 1시간 이내
                }

                return features

        except Exception as e:
            self.logger.error(f"Error processing temporal data: {e}")
            return data if operation == "preprocess" else {}

    async def _process_spatial(self, data: Any, operation: str) -> Any:
        """공간 데이터 처리"""
        try:
            if operation == "preprocess":
                # 공간 데이터 정규화
                if isinstance(data, dict):
                    spatial_data = {}
                    for key, value in data.items():
                        if key.lower() in ['x', 'y', 'z', 'lat', 'lon', 'latitude', 'longitude']:
                            try:
                                spatial_data[key.lower()] = float(value)
                            except:
                                spatial_data[key.lower()] = 0.0
                    return spatial_data
                elif isinstance(data, (list, tuple)) and len(data) >= 2:
                    return {
                        'x': float(data[0]) if len(data) > 0 else 0.0,
                        'y': float(data[1]) if len(data) > 1 else 0.0,
                        'z': float(data[2]) if len(data) > 2 else 0.0
                    }
                else:
                    return {'x': 0.0, 'y': 0.0, 'z': 0.0}

            elif operation == "extract_features":
                if isinstance(data, dict):
                    features = {
                        "dimension_count": len([k for k in data.keys() if k in ['x', 'y', 'z']]),
                        "has_coordinates": bool(set(data.keys()) & {'x', 'y', 'z', 'lat', 'lon'}),
                        "is_2d": 'x' in data and 'y' in data and 'z' not in data,
                        "is_3d": 'x' in data and 'y' in data and 'z' in data
                    }

                    # 거리 계산 (원점에서)
                    if 'x' in data and 'y' in data:
                        distance = (data['x']**2 + data['y']**2)**0.5
                        if 'z' in data:
                            distance = (data['x']**2 + data['y']**2 + data['z']**2)**0.5
                        features["distance_from_origin"] = distance

                    return features

                return {}

        except Exception as e:
            self.logger.error(f"Error processing spatial data: {e}")
            return data if operation == "preprocess" else {}

    async def _process_structural(self, data: Any, operation: str) -> Any:
        """구조 데이터 처리"""
        try:
            if operation == "preprocess":
                # 구조 데이터 정규화
                if isinstance(data, dict):
                    return data
                elif isinstance(data, (list, tuple)):
                    return list(data)
                elif isinstance(data, str):
                    try:
                        # JSON 파싱 시도
                        return json.loads(data)
                    except:
                        return {"raw_text": data}
                else:
                    return {"value": str(data)}

            elif operation == "extract_features":
                features = {"data_type": type(data).__name__}

                if isinstance(data, dict):
                    features.update({
                        "key_count": len(data),
                        "nested_levels": self._calculate_nesting_depth(data),
                        "has_nested_structures": any(isinstance(v, (dict, list)) for v in data.values()),
                        "value_types": list(set(type(v).__name__ for v in data.values()))
                    })

                elif isinstance(data, (list, tuple)):
                    features.update({
                        "length": len(data),
                        "element_types": list(set(type(x).__name__ for x in data)),
                        "is_homogeneous": len(set(type(x).__name__ for x in data)) <= 1,
                        "has_nested_structures": any(isinstance(x, (dict, list)) for x in data)
                    })

                return features

        except Exception as e:
            self.logger.error(f"Error processing structural data: {e}")
            return data if operation == "preprocess" else {}

    async def _process_semantic(self, data: Any, operation: str) -> Any:
        """의미 데이터 처리"""
        try:
            if operation == "preprocess":
                # 의미 데이터 정규화
                if isinstance(data, str):
                    # 의미 분석을 위한 전처리
                    cleaned = re.sub(r'[^\w\s]', ' ', data.lower())
                    return ' '.join(cleaned.split())
                else:
                    return str(data).lower()

            elif operation == "extract_features":
                if isinstance(data, str):
                    words = data.split()

                    # 기본 의미 범주 감지
                    semantic_categories = {
                        "time_related": ["time", "day", "hour", "minute", "today", "yesterday", "tomorrow"],
                        "location_related": ["place", "location", "here", "there", "where"],
                        "action_related": ["do", "make", "create", "build", "run", "execute"],
                        "object_related": ["thing", "object", "item", "element"],
                        "quality_related": ["good", "bad", "big", "small", "fast", "slow"]
                    }

                    detected_categories = []
                    for category, keywords in semantic_categories.items():
                        if any(keyword in data for keyword in keywords):
                            detected_categories.append(category)

                    features = {
                        "semantic_categories": detected_categories,
                        "abstract_score": self._calculate_abstractness(data),
                        "emotional_valence": self._detect_emotional_valence(data),
                        "complexity_level": self._calculate_semantic_complexity(data),
                        "entity_count": len([w for w in words if w.istitle()]),  # 간단한 개체명 감지
                        "concept_density": len(set(words)) / max(1, len(words))
                    }

                    return features

                return {}

        except Exception as e:
            self.logger.error(f"Error processing semantic data: {e}")
            return data if operation == "preprocess" else {}

    async def _default_preprocess(self, data: Any, modality: SensoryModality) -> Any:
        """기본 전처리"""
        try:
            if isinstance(data, str):
                return data.strip()
            elif isinstance(data, (list, tuple)):
                return list(data)
            elif isinstance(data, dict):
                return data
            else:
                return str(data)

        except Exception as e:
            self.logger.error(f"Error in default preprocessing: {e}")
            return data

    async def _default_feature_extraction(self, data: Any, modality: SensoryModality) -> Dict[str, Any]:
        """기본 특징 추출"""
        try:
            features = {
                "data_type": type(data).__name__,
                "modality": modality.value,
                "timestamp": time.time()
            }

            if isinstance(data, str):
                features["length"] = len(data)
                features["is_empty"] = len(data.strip()) == 0

            elif isinstance(data, (list, tuple)):
                features["length"] = len(data)
                features["is_empty"] = len(data) == 0

            elif isinstance(data, dict):
                features["key_count"] = len(data)
                features["is_empty"] = len(data) == 0

            return features

        except Exception as e:
            self.logger.error(f"Error in default feature extraction: {e}")
            return {}

    async def _assess_quality(self, data: Any, features: Dict[str, Any],
                            modality: SensoryModality) -> Dict[str, float]:
        """데이터 품질 평가"""
        try:
            quality_metrics = {}

            # 기본 품질 지표
            quality_metrics["completeness"] = 1.0 if data else 0.0

            # 모달리티별 품질 평가
            if modality == SensoryModality.TEXTUAL:
                quality_metrics["readability"] = self._assess_text_readability(data)
                quality_metrics["coherence"] = self._assess_text_coherence(data)

            elif modality == SensoryModality.NUMERICAL:
                quality_metrics["validity"] = self._assess_numerical_validity(data)
                quality_metrics["precision"] = self._assess_numerical_precision(data)

            elif modality == SensoryModality.STRUCTURAL:
                quality_metrics["consistency"] = self._assess_structural_consistency(data)
                quality_metrics["completeness"] = self._assess_structural_completeness(data)

            # 전체 품질 점수 계산
            if quality_metrics:
                quality_metrics["overall_quality"] = sum(quality_metrics.values()) / len(quality_metrics)
            else:
                quality_metrics["overall_quality"] = 0.5

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Error assessing quality: {e}")
            return {"overall_quality": 0.5}

    async def _normalize_data(self, data: Any, modality: SensoryModality) -> Any:
        """데이터 정규화"""
        try:
            if modality == SensoryModality.NUMERICAL and isinstance(data, (list, tuple)):
                # 수치 데이터 정규화 (0-1 범위)
                if data and all(isinstance(x, (int, float)) for x in data):
                    min_val, max_val = min(data), max(data)
                    if max_val != min_val:
                        return [(x - min_val) / (max_val - min_val) for x in data]

            return data

        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return data

    def _calculate_text_complexity(self, text: str) -> float:
        """텍스트 복잡도 계산"""
        try:
            if not text:
                return 0.0

            words = text.split()
            sentences = [s for s in text.split('.') if s.strip()]

            if not words or not sentences:
                return 0.0

            avg_word_length = sum(len(word) for word in words) / len(words)
            avg_sentence_length = len(words) / len(sentences)

            # 간단한 복잡도 지표
            complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.7) / 20
            return min(1.0, complexity)

        except Exception as e:
            self.logger.error(f"Error calculating text complexity: {e}")
            return 0.5

    def _calculate_variance(self, numbers: List[float]) -> float:
        """분산 계산"""
        try:
            if len(numbers) < 2:
                return 0.0

            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            return variance

        except Exception as e:
            self.logger.error(f"Error calculating variance: {e}")
            return 0.0

    def _is_numeric_sequence(self, numbers: List[float]) -> bool:
        """수치 시퀀스 여부 확인"""
        try:
            if len(numbers) < 3:
                return False

            # 등차수열 확인
            diff = numbers[1] - numbers[0]
            for i in range(2, len(numbers)):
                if abs((numbers[i] - numbers[i-1]) - diff) > 0.01:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking numeric sequence: {e}")
            return False

    def _detect_trend(self, numbers: List[float]) -> str:
        """트렌드 감지"""
        try:
            if len(numbers) < 2:
                return "stable"

            increases = sum(1 for i in range(1, len(numbers)) if numbers[i] > numbers[i-1])
            decreases = sum(1 for i in range(1, len(numbers)) if numbers[i] < numbers[i-1])

            if increases > decreases * 1.5:
                return "increasing"
            elif decreases > increases * 1.5:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            self.logger.error(f"Error detecting trend: {e}")
            return "unknown"

    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """중첩 깊이 계산"""
        try:
            if isinstance(data, dict):
                if not data:
                    return current_depth
                max_depth = current_depth
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        depth = self._calculate_nesting_depth(value, current_depth + 1)
                        max_depth = max(max_depth, depth)
                return max_depth

            elif isinstance(data, list):
                if not data:
                    return current_depth
                max_depth = current_depth
                for item in data:
                    if isinstance(item, (dict, list)):
                        depth = self._calculate_nesting_depth(item, current_depth + 1)
                        max_depth = max(max_depth, depth)
                return max_depth

            return current_depth

        except Exception as e:
            self.logger.error(f"Error calculating nesting depth: {e}")
            return current_depth

    def _calculate_abstractness(self, text: str) -> float:
        """추상성 계산"""
        try:
            abstract_words = ["idea", "concept", "thought", "feeling", "belief", "theory", "principle"]
            concrete_words = ["object", "thing", "item", "tool", "machine", "building", "person"]

            words = text.lower().split()
            abstract_count = sum(1 for word in words if word in abstract_words)
            concrete_count = sum(1 for word in words if word in concrete_words)

            if abstract_count + concrete_count == 0:
                return 0.5

            return abstract_count / (abstract_count + concrete_count)

        except Exception as e:
            self.logger.error(f"Error calculating abstractness: {e}")
            return 0.5

    def _detect_emotional_valence(self, text: str) -> str:
        """감정적 가치 감지"""
        try:
            positive_words = ["good", "great", "excellent", "happy", "joy", "love", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "sad", "hate", "anger", "horrible"]

            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)

            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error detecting emotional valence: {e}")
            return "neutral"

    def _calculate_semantic_complexity(self, text: str) -> float:
        """의미적 복잡도 계산"""
        try:
            words = text.split()
            unique_words = set(words)

            if not words:
                return 0.0

            # 어휘 다양성
            lexical_diversity = len(unique_words) / len(words)

            # 평균 단어 길이
            avg_word_length = sum(len(word) for word in words) / len(words)

            # 복잡도 조합
            complexity = (lexical_diversity * 0.6 + min(avg_word_length / 10, 1.0) * 0.4)
            return complexity

        except Exception as e:
            self.logger.error(f"Error calculating semantic complexity: {e}")
            return 0.5

    def _assess_text_readability(self, text: str) -> float:
        """텍스트 가독성 평가"""
        try:
            if not isinstance(text, str) or not text.strip():
                return 0.0

            words = text.split()
            sentences = [s for s in text.split('.') if s.strip()]

            if not words or not sentences:
                return 0.0

            avg_sentence_length = len(words) / len(sentences)

            # 간단한 가독성 지표 (짧은 문장이 더 읽기 쉬움)
            readability = max(0.0, 1.0 - (avg_sentence_length - 10) / 20)
            return min(1.0, readability)

        except Exception as e:
            self.logger.error(f"Error assessing text readability: {e}")
            return 0.5

    def _assess_text_coherence(self, text: str) -> float:
        """텍스트 일관성 평가"""
        try:
            if not isinstance(text, str) or not text.strip():
                return 0.0

            sentences = [s.strip() for s in text.split('.') if s.strip()]

            if len(sentences) < 2:
                return 1.0

            # 간단한 일관성 지표 (문장 길이의 일관성)
            sentence_lengths = [len(s.split()) for s in sentences]
            if not sentence_lengths:
                return 0.0

            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)

            # 분산이 작을수록 일관성이 높음
            coherence = max(0.0, 1.0 - variance / 100)
            return min(1.0, coherence)

        except Exception as e:
            self.logger.error(f"Error assessing text coherence: {e}")
            return 0.5

    def _assess_numerical_validity(self, data: Any) -> float:
        """수치 데이터 유효성 평가"""
        try:
            if isinstance(data, (int, float)):
                return 1.0 if not (data != data or data == float('inf') or data == float('-inf')) else 0.0

            elif isinstance(data, (list, tuple)):
                if not data:
                    return 0.0

                valid_count = sum(
                    1 for x in data
                    if isinstance(x, (int, float)) and not (x != x or x == float('inf') or x == float('-inf'))
                )
                return valid_count / len(data)

            return 0.0

        except Exception as e:
            self.logger.error(f"Error assessing numerical validity: {e}")
            return 0.0

    def _assess_numerical_precision(self, data: Any) -> float:
        """수치 데이터 정밀도 평가"""
        try:
            if isinstance(data, int):
                return 1.0
            elif isinstance(data, float):
                # 소수점 자릿수 확인
                decimal_places = len(str(data).split('.')[-1]) if '.' in str(data) else 0
                return max(0.5, 1.0 - decimal_places / 10)

            elif isinstance(data, (list, tuple)):
                if not data:
                    return 0.0

                precisions = []
                for x in data:
                    if isinstance(x, int):
                        precisions.append(1.0)
                    elif isinstance(x, float):
                        decimal_places = len(str(x).split('.')[-1]) if '.' in str(x) else 0
                        precisions.append(max(0.5, 1.0 - decimal_places / 10))

                return sum(precisions) / len(precisions) if precisions else 0.0

            return 0.5

        except Exception as e:
            self.logger.error(f"Error assessing numerical precision: {e}")
            return 0.5

    def _assess_structural_consistency(self, data: Any) -> float:
        """구조 데이터 일관성 평가"""
        try:
            if isinstance(data, dict):
                if not data:
                    return 1.0

                # 모든 값의 타입 일관성 확인
                value_types = [type(v).__name__ for v in data.values()]
                unique_types = set(value_types)

                consistency = 1.0 - (len(unique_types) - 1) / max(1, len(value_types))
                return max(0.0, consistency)

            elif isinstance(data, (list, tuple)):
                if not data:
                    return 1.0

                element_types = [type(x).__name__ for x in data]
                unique_types = set(element_types)

                consistency = 1.0 - (len(unique_types) - 1) / max(1, len(element_types))
                return max(0.0, consistency)

            return 1.0

        except Exception as e:
            self.logger.error(f"Error assessing structural consistency: {e}")
            return 0.5

    def _assess_structural_completeness(self, data: Any) -> float:
        """구조 데이터 완전성 평가"""
        try:
            if isinstance(data, dict):
                if not data:
                    return 0.0

                # None 값이나 빈 값의 비율 확인
                empty_count = sum(1 for v in data.values() if v is None or v == "")
                completeness = 1.0 - empty_count / len(data)
                return max(0.0, completeness)

            elif isinstance(data, (list, tuple)):
                if not data:
                    return 0.0

                empty_count = sum(1 for x in data if x is None or x == "")
                completeness = 1.0 - empty_count / len(data)
                return max(0.0, completeness)

            return 1.0

        except Exception as e:
            self.logger.error(f"Error assessing structural completeness: {e}")
            return 0.5

    def _update_metrics(self, result: SensoryResult) -> None:
        """메트릭 업데이트"""
        self._total_processed += 1

        if result.success:
            self._successful_processed += 1

        if result.processing_time_ms > 0:
            self._processing_times.append(result.processing_time_ms)
            # 최근 100개 기록만 유지
            if len(self._processing_times) > 100:
                self._processing_times = self._processing_times[-50:]


async def create_sensory_processor() -> SensoryProcessor:
    """SensoryProcessor 인스턴스 생성 및 초기화"""
    processor = SensoryProcessor()
    await processor.initialize()
    return processor