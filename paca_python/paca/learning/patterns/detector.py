"""
Pattern Detector
학습 패턴 감지 시스템
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ...core.types import Result, create_success, create_failure
from ..auto.types import LearningPattern, PatternType

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """패턴 감지 결과"""
    pattern: LearningPattern
    confidence: float
    matched_keywords: List[str]
    matched_contexts: List[str]
    text_location: Tuple[int, int]  # 시작, 끝 위치


class PatternDetector:
    """
    학습 패턴 감지 시스템
    텍스트에서 학습 기회를 자동으로 감지
    """

    def __init__(self):
        self.korean_stopwords = {
            "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
            "와", "과", "하고", "도", "만", "부터", "까지", "한테", "에게"
        }

    def detect_patterns(
        self,
        text: str,
        patterns: List[LearningPattern],
        min_confidence: float = 0.5
    ) -> Result[List[DetectionResult]]:
        """텍스트에서 패턴 감지"""
        try:
            results = []
            normalized_text = self._normalize_text(text)

            for pattern in patterns:
                detection_result = self._detect_single_pattern(normalized_text, pattern)
                if detection_result and detection_result.confidence >= min_confidence:
                    results.append(detection_result)

            # 신뢰도 순으로 정렬
            results.sort(key=lambda x: x.confidence, reverse=True)

            return create_success(results)

        except Exception as e:
            return create_failure(f"Pattern detection failed: {str(e)}")

    def _detect_single_pattern(
        self,
        text: str,
        pattern: LearningPattern
    ) -> Optional[DetectionResult]:
        """단일 패턴 감지"""
        matched_keywords = []
        matched_contexts = []

        # 키워드 매칭
        for keyword in pattern.keywords:
            if self._find_keyword_in_text(text, keyword):
                matched_keywords.append(keyword)

        # 컨텍스트 지시자 매칭
        for context in pattern.context_indicators:
            if self._find_keyword_in_text(text, context):
                matched_contexts.append(context)

        # 신뢰도 계산
        confidence = self._calculate_pattern_confidence(
            matched_keywords, matched_contexts, pattern, text
        )

        if confidence < pattern.confidence_threshold * 0.3:  # 최소 임계값
            return None

        # 텍스트 위치 찾기
        text_location = self._find_pattern_location(text, matched_keywords + matched_contexts)

        return DetectionResult(
            pattern=pattern,
            confidence=confidence,
            matched_keywords=matched_keywords,
            matched_contexts=matched_contexts,
            text_location=text_location
        )

    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 소문자 변환
        text = text.lower()

        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()

        # 특수 문자 처리 (한국어 고려)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)

        return text

    def _find_keyword_in_text(self, text: str, keyword: str) -> bool:
        """텍스트에서 키워드 찾기 (한국어 어미 변화 고려)"""
        keyword = keyword.lower()

        # 정확한 매칭
        if keyword in text:
            return True

        # 한국어 어미 변화 고려한 매칭
        if self._korean_stem_match(text, keyword):
            return True

        return False

    def _korean_stem_match(self, text: str, keyword: str) -> bool:
        """한국어 어간 매칭"""
        # 간단한 어간 추출 (실제로는 더 정교한 형태소 분석 필요)
        if len(keyword) < 2:
            return False

        # 어간 추출 (마지막 1-2글자 제거)
        stem1 = keyword[:-1]
        stem2 = keyword[:-2] if len(keyword) > 2 else keyword

        return stem1 in text or stem2 in text

    def _calculate_pattern_confidence(
        self,
        matched_keywords: List[str],
        matched_contexts: List[str],
        pattern: LearningPattern,
        text: str
    ) -> float:
        """패턴 신뢰도 계산"""
        # 기본 점수
        base_score = 0.3

        # 키워드 매칭 점수 (40%)
        keyword_score = len(matched_keywords) / max(len(pattern.keywords), 1)
        keyword_score = min(keyword_score, 1.0) * 0.4

        # 컨텍스트 매칭 점수 (40%)
        context_score = len(matched_contexts) / max(len(pattern.context_indicators), 1)
        context_score = min(context_score, 1.0) * 0.4

        # 텍스트 길이 보너스 (10%)
        length_bonus = min(len(text) / 100.0, 1.0) * 0.1

        # 패턴 가중치 적용 (10%)
        weight_bonus = (pattern.weight - 1.0) * 0.1

        total_score = base_score + keyword_score + context_score + length_bonus + weight_bonus

        return min(total_score, 1.0)

    def _find_pattern_location(
        self,
        text: str,
        matched_terms: List[str]
    ) -> Tuple[int, int]:
        """패턴이 감지된 텍스트 위치 찾기"""
        if not matched_terms:
            return (0, len(text))

        # 첫 번째와 마지막 매칭 위치 찾기
        positions = []
        for term in matched_terms:
            pos = text.find(term.lower())
            if pos != -1:
                positions.append((pos, pos + len(term)))

        if not positions:
            return (0, len(text))

        start_pos = min(pos[0] for pos in positions)
        end_pos = max(pos[1] for pos in positions)

        return (start_pos, end_pos)

    def extract_context_around_pattern(
        self,
        text: str,
        detection_result: DetectionResult,
        context_window: int = 50
    ) -> str:
        """패턴 주변 컨텍스트 추출"""
        start, end = detection_result.text_location

        # 컨텍스트 윈도우 적용
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)

        context = text[context_start:context_end]

        # 단어 경계에서 자르기
        if context_start > 0:
            first_space = context.find(' ')
            if first_space != -1:
                context = context[first_space + 1:]

        if context_end < len(text):
            last_space = context.rfind(' ')
            if last_space != -1:
                context = context[:last_space]

        return context.strip()

    def get_pattern_statistics(
        self,
        detection_results: List[DetectionResult]
    ) -> Dict[str, Any]:
        """패턴 감지 통계"""
        if not detection_results:
            return {"total_patterns": 0}

        pattern_counts = {}
        confidence_sum = 0.0

        for result in detection_results:
            pattern_type = result.pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            confidence_sum += result.confidence

        return {
            "total_patterns": len(detection_results),
            "pattern_counts": pattern_counts,
            "average_confidence": confidence_sum / len(detection_results),
            "highest_confidence": max(r.confidence for r in detection_results),
            "dominant_pattern": max(pattern_counts.items(), key=lambda x: x[1])[0]
        }