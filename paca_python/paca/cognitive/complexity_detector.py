"""
복잡도 감지 시스템 (Complexity Detection System)
PACA v5 Python의 핵심 차별화 기능

사용자 질문의 복잡도를 0-100점으로 자동 평가하여
추론 체인 활성화 여부를 결정하는 시스템
"""

import json
import logging
import re
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from paca.core.types import current_timestamp, generate_id
from paca.core.validators import is_valid_korean_text, validate_string_length

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """질문 도메인 유형"""

    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"


class ComplexityLevel(Enum):
    """복잡도 수준"""

    SIMPLE = (0, 20)
    MODERATE = (21, 40)
    COMPLEX = (41, 70)
    VERY_COMPLEX = (71, 90)
    EXTREME = (91, 100)


@dataclass
class ComplexityResult:
    """복잡도 감지 결과"""

    score: int
    reasoning_required: bool
    domain: DomainType
    confidence: float
    level: ComplexityLevel
    analysis_details: Dict[str, Any]
    processing_time_ms: float
    timestamp: float
    unique_id: str


@dataclass
class ComplexityMetrics:
    """복잡도 분석 메트릭"""

    keyword_score: float = 0.0
    structure_score: float = 0.0
    domain_score: float = 0.0
    reasoning_score: float = 0.0

    def calculate_total(self, weights: Optional[Dict[str, float]] = None) -> int:
        weights = weights or {
            'keyword': 0.30,
            'structure': 0.25,
            'domain': 0.25,
            'reasoning': 0.20,
        }

        total = (
            self.keyword_score * weights.get('keyword', 0.30)
            + self.structure_score * weights.get('structure', 0.25)
            + self.domain_score * weights.get('domain', 0.25)
            + self.reasoning_score * weights.get('reasoning', 0.20)
        )
        return min(100, max(0, int(total)))


@dataclass
class ComplexityFeatureSet:
    """복잡도 분석에 사용되는 피처 모음"""

    text: str
    normalized_text: str
    character_length: int
    sentence_count: int
    clause_count: int
    question_count: int
    conditional_count: int
    analysis_term_count: int
    domain_hits: Dict[DomainType, int]
    matched_keywords: Dict[DomainType, List[str]]
    technical_terms: int
    honorific_count: int
    average_sentence_length: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'character_length': self.character_length,
            'sentence_count': self.sentence_count,
            'clause_count': self.clause_count,
            'question_count': self.question_count,
            'conditional_count': self.conditional_count,
            'analysis_term_count': self.analysis_term_count,
            'technical_terms': self.technical_terms,
            'honorific_count': self.honorific_count,
            'average_sentence_length': round(self.average_sentence_length, 2),
            'domain_hits': {domain.value: count for domain, count in self.domain_hits.items()},
            'matched_keywords': {
                domain.value: keywords for domain, keywords in self.matched_keywords.items()
            },
        }


class ComplexityDetector:
    """텍스트 복잡도를 정량화하는 감지기"""

    CONFIG_FILENAME = "complexity_thresholds.json"
    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent / "data" / "config" / CONFIG_FILENAME
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_configuration(config or {})
        complexity_settings = self.config.get('complexity', self.config)

        self.reasoning_threshold = int(complexity_settings.get('reasoning_threshold', 45))
        self.feature_weights = self._normalize_weights(
            complexity_settings.get(
                'feature_weights',
                {'keyword': 0.35, 'structure': 0.25, 'domain': 0.20, 'reasoning': 0.20},
            )
        )
        self.level_thresholds = self._prepare_level_thresholds(
            complexity_settings.get('level_thresholds')
        )
        self.pattern_weights = complexity_settings.get(
            'pattern_weights',
            {
                'multi_question': 18,
                'conditional': 12,
                'comparison': 14,
                'sequence': 10,
                'analysis': 16,
                'complex_grammar': 12,
                'technical_terms': 14,
                'korean_honorific': -6,
            },
        )
        self.reasoning_indicators = complexity_settings.get(
            'reasoning_indicators',
            {
                '왜': 25,
                '어떻게': 20,
                '원인': 22,
                '결과': 18,
                '이유': 22,
                '방법': 15,
                '과정': 15,
                '단계': 12,
                '분석': 20,
                '예측': 18,
                '추론': 24,
                '결론': 20,
            },
        )
        self.question_tokens = complexity_settings.get(
            'question_tokens', ['?', '？', '무엇', '왜', '어떻게', '궁금', '알고싶', '모르겠']
        )

        cache_settings = complexity_settings.get('cache', {})
        self.cache_enabled = cache_settings.get('enabled', True)
        self.cache_ttl = cache_settings.get('ttl_seconds', 300)
        self.max_cache_size = cache_settings.get('max_entries', 50)
        self._cache: OrderedDict[str, Tuple[ComplexityResult, float]] = OrderedDict()

        self.config_source = complexity_settings.get('__source__', str(self.DEFAULT_CONFIG_PATH))

        self._init_domain_keywords(complexity_settings.get('domain_keywords'))
        self._init_complexity_patterns(complexity_settings.get('regex_patterns'))

        self.analysis_count = 0
        self.total_processing_time = 0.0

        logger.info(
            "ComplexityDetector 초기화 완료",
            extra={'reasoning_threshold': self.reasoning_threshold, 'config_source': self.config_source},
        )

    def _load_configuration(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        file_config: Dict[str, Any] = {}
        if self.DEFAULT_CONFIG_PATH.exists():
            try:
                with self.DEFAULT_CONFIG_PATH.open('r', encoding='utf-8') as handle:
                    file_config = json.load(handle)
                    file_config.setdefault('complexity', {})['__source__'] = str(self.DEFAULT_CONFIG_PATH)
            except json.JSONDecodeError as error:
                logger.warning("ComplexityDetector 설정 파일 파싱 실패", extra={'error': str(error)})
        return self._deep_merge_dicts(file_config, overrides)

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = ComplexityDetector._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values()) or 1.0
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def _prepare_level_thresholds(
        self, overrides: Optional[Dict[str, Union[List[int], Tuple[int, int]]]]
    ) -> Dict[ComplexityLevel, Tuple[int, int]]:
        thresholds = {level: level.value for level in ComplexityLevel}
        if not overrides:
            return thresholds

        for name, bounds in overrides.items():
            try:
                level = ComplexityLevel[name]
            except KeyError:
                continue

            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                thresholds[level] = (int(bounds[0]), int(bounds[1]))
        return thresholds

    def _init_domain_keywords(self, overrides: Optional[Dict[str, Any]]) -> None:
        base_keywords = {
            DomainType.MATHEMATICAL: {
                'keywords': ['계산', '수학', '공식', '증명', '방정식', '함수', '미적분', '통계', '확률', '알고리즘'],
                'complexity_weight': 1.2,
            },
            DomainType.LOGICAL: {
                'keywords': ['논리', '추론', '증명', '결론', '전제', '가설', '분석', '왜', '어떻게', '원인'],
                'complexity_weight': 1.3,
            },
            DomainType.CREATIVE: {
                'keywords': ['창작', '아이디어', '상상', '디자인', '예술', '시', '소설', '음악', '발명'],
                'complexity_weight': 1.05,
            },
            DomainType.ANALYTICAL: {
                'keywords': ['분석', '해석', '평가', '비교', '검토', '조사', '연구', '데이터', '통계'],
                'complexity_weight': 1.4,
            },
            DomainType.TECHNICAL: {
                'keywords': ['프로그래밍', '코딩', '개발', '시스템', '네트워크', '데이터베이스', 'API', '알고리즘'],
                'complexity_weight': 1.35,
            },
            DomainType.FACTUAL: {
                'keywords': ['무엇', '언제', '어디서', '누구', '정보', '사실', '역사', '정의'],
                'complexity_weight': 0.8,
            },
            DomainType.CONVERSATIONAL: {
                'keywords': ['안녕', '고마워', '미안', '어떻게', '잘지내', '날씨', '기분'],
                'complexity_weight': 0.5,
            },
            DomainType.EMOTIONAL: {
                'keywords': ['감정', '기분', '우울', '행복', '스트레스', '고민', '상담', '도움'],
                'complexity_weight': 0.95,
            },
        }

        overrides = overrides or {}
        mapped: Dict[DomainType, Dict[str, Any]] = {}

        for domain, defaults in base_keywords.items():
            override = overrides.get(domain.value) or overrides.get(domain.name)
            if override:
                mapped[domain] = {
                    'keywords': override.get('keywords', defaults['keywords']),
                    'complexity_weight': float(override.get('complexity_weight', defaults['complexity_weight'])),
                }
            else:
                mapped[domain] = defaults

        self.domain_keywords = mapped

    def _init_complexity_patterns(self, overrides: Optional[Dict[str, str]]) -> None:
        base_patterns = {
            'multi_question': r'[?？]{2,}|그리고.*[?？]|또.*[?？]',
            'conditional': r'만약|만일|가정|경우|조건',
            'comparison': r'비교|차이|대비|반대|versus|vs',
            'sequence': r'단계|순서|과정|절차|방법|어떻게',
            'analysis': r'분석|해석|평가|검토|이유|원인|결과',
            'complex_grammar': r'(?:.*이.*이며.*이고)|(?:.*하면서.*하고.*한다)',
            'technical_terms': r'[A-Z]{2,}|[a-z]+[A-Z][a-z]*',
            'korean_honorific': r'습니다|세요|시오|십시오',
        }

        if overrides:
            base_patterns.update(overrides)
        self.complexity_patterns = base_patterns

    async def detect_complexity(self, user_input: str) -> ComplexityResult:
        start_time = time.time()
        normalized_input = (user_input or '').strip()

        if not validate_string_length(normalized_input, 1, 10000):
            return self._create_error_result("입력 텍스트 길이가 범위를 벗어났습니다")

        if self.cache_enabled:
            cached = self._get_cached_result(normalized_input)
            if cached:
                return cached

        try:
            features = await self.extract_features(normalized_input)
            score, level, reasons, metrics = self.score_features(features)
            domain, domain_confidence = self._detect_domain_from_features(features)
            reasoning_required = self.should_use_reasoning_chain(score)
            processing_time = (time.time() - start_time) * 1000

            analysis_details = {
                'features': features.to_dict(),
                'metrics': {
                    'keyword_score': round(metrics.keyword_score, 2),
                    'structure_score': round(metrics.structure_score, 2),
                    'domain_score': round(metrics.domain_score, 2),
                    'reasoning_score': round(metrics.reasoning_score, 2),
                },
                'weights': self.feature_weights,
                'reasons': reasons,
                'reasoning_threshold': self.reasoning_threshold,
                'config_source': self.config_source,
                'level_thresholds': {
                    level.name: bounds for level, bounds in self.level_thresholds.items()
                },
                'cache_hit': False,
                'korean_text': is_valid_korean_text(normalized_input),
            }

            result = ComplexityResult(
                score=score,
                reasoning_required=reasoning_required,
                domain=domain,
                confidence=round(domain_confidence, 2),
                level=level,
                analysis_details=analysis_details,
                processing_time_ms=processing_time,
                timestamp=current_timestamp(),
                unique_id=generate_id('complexity_'),
            )

            self._cache_result(normalized_input, result)
            self._update_performance_metrics(processing_time)

            logger.debug(
                "복잡도 감지 완료",
                extra={
                    'score': score,
                    'level': level.name,
                    'domain': domain.value,
                    'processing_time_ms': round(processing_time, 2),
                },
            )

            return result

        except Exception as error:
            logger.error("복잡도 감지 중 오류 발생", exc_info=error)
            return self._create_error_result(f"분석 오류: {error}")

    async def extract_features(self, text: str) -> ComplexityFeatureSet:
        normalized = re.sub(r'\s+', ' ', text)
        lower_text = normalized.lower()
        sentences = [s for s in re.split(r'[.!?。！？]', normalized) if s.strip()]

        character_length = len(normalized)
        sentence_count = len(sentences)
        clause_count = len(re.findall(r'그리고|그러나|하지만|또는|따라서|때문에', normalized))

        question_count = 0
        for token in self.question_tokens:
            if token.strip() in {'?', '？'}:
                question_count += normalized.count(token.strip())
            else:
                question_count += lower_text.count(token)

        conditional_count = len(re.findall(self.complexity_patterns['conditional'], lower_text))
        analysis_term_count = len(re.findall(self.complexity_patterns['analysis'], lower_text))

        domain_hits: Dict[DomainType, int] = {}
        matched_keywords: Dict[DomainType, List[str]] = {}
        for domain, data in self.domain_keywords.items():
            keywords = data.get('keywords', [])
            matches = [kw for kw in keywords if kw.lower() in lower_text]
            if matches:
                matched_keywords[domain] = matches[:5]
            domain_hits[domain] = len(matches)

        technical_terms = len(re.findall(self.complexity_patterns['technical_terms'], text))
        honorific_count = len(re.findall(self.complexity_patterns['korean_honorific'], text))

        average_sentence_length = (
            character_length / sentence_count if sentence_count else character_length
        )

        return ComplexityFeatureSet(
            text=text,
            normalized_text=normalized,
            character_length=character_length,
            sentence_count=sentence_count,
            clause_count=clause_count,
            question_count=question_count,
            conditional_count=conditional_count,
            analysis_term_count=analysis_term_count,
            domain_hits=domain_hits,
            matched_keywords=matched_keywords,
            technical_terms=technical_terms,
            honorific_count=honorific_count,
            average_sentence_length=average_sentence_length,
        )

    def score_features(
        self, features: ComplexityFeatureSet
    ) -> Tuple[int, ComplexityLevel, List[str], ComplexityMetrics]:
        metrics = ComplexityMetrics()
        reasons: List[str] = []

        metrics.keyword_score, keyword_reasons = self._score_keyword_component(features)
        metrics.structure_score, structure_reasons = self._score_structure_component(features)
        metrics.domain_score, domain_reasons = self._score_domain_component(features)
        metrics.reasoning_score, reasoning_reasons = self._score_reasoning_component(features)

        reasons.extend(keyword_reasons)
        reasons.extend(structure_reasons)
        reasons.extend(domain_reasons)
        reasons.extend(reasoning_reasons)

        total_score = metrics.calculate_total(self.feature_weights)
        level = self._determine_level(total_score)

        return total_score, level, reasons, metrics

    def _score_keyword_component(self, features: ComplexityFeatureSet) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        for domain, hit_count in features.domain_hits.items():
            if hit_count <= 0:
                continue
            weight = self.domain_keywords.get(domain, {}).get('complexity_weight', 1.0)
            increment = min(60.0, hit_count * weight * 12)
            score += increment
            reasons.append(f"{domain.value} 키워드 {hit_count}개 감지(+{int(increment)})")

        for pattern_name, pattern in self.complexity_patterns.items():
            if pattern_name == 'korean_honorific':
                continue
            if re.search(pattern, features.normalized_text, re.IGNORECASE):
                delta = self.pattern_weights.get(pattern_name, 10)
                if delta > 0:
                    score += delta
                    reasons.append(f"패턴 '{pattern_name}' 감지(+{delta})")

        if features.honorific_count:
            penalty = abs(self.pattern_weights.get('korean_honorific', -6)) * features.honorific_count
            score = max(0.0, score - penalty)
            reasons.append(f"높임말 {features.honorific_count}회 (-{penalty})")

        if features.technical_terms:
            bonus = min(20.0, features.technical_terms * 4)
            score += bonus
            reasons.append(f"기술 용어 {features.technical_terms}개(+{int(bonus)})")

        return min(100.0, score), reasons

    def _score_structure_component(self, features: ComplexityFeatureSet) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        length_score = min(35.0, features.character_length / 20)
        if length_score > 0:
            score += length_score
            reasons.append(f"문장 길이 기반 점수 {int(length_score)}")

        sentence_score = min(25.0, features.sentence_count * 5)
        if sentence_score:
            score += sentence_score
            reasons.append(f"문장 수 {features.sentence_count}개(+{int(sentence_score)})")

        clause_score = min(20.0, features.clause_count * 4)
        if clause_score:
            score += clause_score
            reasons.append(f"절 연접 {features.clause_count}회(+{int(clause_score)})")

        if features.average_sentence_length > 35:
            complexity_bonus = min(20.0, (features.average_sentence_length - 35) / 2)
            score += complexity_bonus
            reasons.append(f"긴 문장 구조 감지(+{int(complexity_bonus)})")

        return min(100.0, score), reasons

    def _score_domain_component(self, features: ComplexityFeatureSet) -> Tuple[float, List[str]]:
        domain, confidence = self._detect_domain_from_features(features)
        base_score = 40.0
        if domain in self.domain_keywords:
            weight = self.domain_keywords[domain].get('complexity_weight', 1.0)
            base_score = 40.0 + (weight - 1.0) * 40.0

        score = min(100.0, base_score * max(confidence, 0.4))
        reasons = [f"도메인 '{domain.value}' 감지(신뢰도 {confidence:.2f})"]
        return score, reasons

    def _score_reasoning_component(self, features: ComplexityFeatureSet) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        lower_text = features.normalized_text.lower()

        for token, weight in self.reasoning_indicators.items():
            occurrences = lower_text.count(token)
            if occurrences:
                increment = weight * occurrences
                score += increment
                reasons.append(f"추론 지표 '{token}' {occurrences}회(+{int(increment)})")

        if features.question_count:
            question_bonus = min(25.0, features.question_count * 5)
            score += question_bonus
            reasons.append(f"질문 패턴 {features.question_count}회(+{int(question_bonus)})")

        if features.analysis_term_count:
            analysis_bonus = min(30.0, features.analysis_term_count * 6)
            score += analysis_bonus
            reasons.append(f"분석 용어 {features.analysis_term_count}회(+{int(analysis_bonus)})")

        if features.conditional_count:
            conditional_bonus = min(20.0, features.conditional_count * 6)
            score += conditional_bonus
            reasons.append(f"조건문 {features.conditional_count}회(+{int(conditional_bonus)})")

        return min(100.0, score), reasons

    def _detect_domain_from_features(
        self, features: ComplexityFeatureSet
    ) -> Tuple[DomainType, float]:
        best_domain = DomainType.CONVERSATIONAL
        best_hits = -1

        for domain, hits in features.domain_hits.items():
            if hits > best_hits:
                best_domain = domain
                best_hits = hits

        if best_hits <= 0:
            return DomainType.CONVERSATIONAL, 0.45

        weight = self.domain_keywords.get(best_domain, {}).get('complexity_weight', 1.0)
        confidence = min(1.0, max(0.3, best_hits * 0.35 * weight))
        return best_domain, confidence

    def _get_cached_result(self, key: str) -> Optional[ComplexityResult]:
        cached = self._cache.get(key)
        if not cached:
            return None

        result, timestamp = cached
        if (time.time() - timestamp) > self.cache_ttl:
            del self._cache[key]
            return None

        self._cache.move_to_end(key)
        updated_details = dict(result.analysis_details)
        updated_details['cache_hit'] = True
        return replace(result, analysis_details=updated_details)

    def _cache_result(self, key: str, result: ComplexityResult) -> None:
        if not self.cache_enabled:
            return

        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def _determine_level(self, score: int) -> ComplexityLevel:
        for level, bounds in self.level_thresholds.items():
            min_score, max_score = bounds
            if min_score <= score <= max_score:
                return level
        return ComplexityLevel.EXTREME

    def should_use_reasoning_chain(self, complexity_score: int) -> bool:
        return complexity_score >= self.reasoning_threshold

    def _create_error_result(self, error_message: str) -> ComplexityResult:
        return ComplexityResult(
            score=0,
            reasoning_required=False,
            domain=DomainType.CONVERSATIONAL,
            confidence=0.0,
            level=ComplexityLevel.SIMPLE,
            analysis_details={'error': error_message, 'cache_hit': False},
            processing_time_ms=0.0,
            timestamp=current_timestamp(),
            unique_id=generate_id('error_'),
        )

    def _update_performance_metrics(self, processing_time: float) -> None:
        self.analysis_count += 1
        self.total_processing_time += processing_time

    def get_performance_stats(self) -> Dict[str, Any]:
        if self.analysis_count == 0:
            return {'average_processing_time_ms': 0, 'total_analyses': 0}

        return {
            'average_processing_time_ms': self.total_processing_time / self.analysis_count,
            'total_analyses': self.analysis_count,
            'total_processing_time_ms': self.total_processing_time,
        }

    def update_threshold(self, new_threshold: int) -> None:
        if 0 <= new_threshold <= 100:
            self.reasoning_threshold = new_threshold
            logger.info("추론 임계값 업데이트", extra={'new_threshold': new_threshold})
        else:
            logger.warning("유효하지 않은 임계값", extra={'new_threshold': new_threshold})


async def detect_complexity(user_input: str, config: Optional[Dict[str, Any]] = None) -> ComplexityResult:
    detector = ComplexityDetector(config)
    return await detector.detect_complexity(user_input)


def create_complexity_detector(reasoning_threshold: int = 30) -> ComplexityDetector:
    config = {'complexity': {'reasoning_threshold': reasoning_threshold}}
    return ComplexityDetector(config)


__all__ = [
    'ComplexityDetector',
    'ComplexityResult',
    'ComplexityMetrics',
    'ComplexityFeatureSet',
    'ComplexityLevel',
    'DomainType',
    'detect_complexity',
    'create_complexity_detector',
]
