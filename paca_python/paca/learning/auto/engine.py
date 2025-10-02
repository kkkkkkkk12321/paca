"""
Auto Learning Engine
자동 학습 시스템 메인 구현
TypeScript 버전의 Python 완전 변환 + 한국어 NLP 최적화
"""

import asyncio
import copy
import json
import time
import logging
from dataclasses import fields
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import re

# Korean NLP
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    print("Warning: KoNLPy not available. Korean NLP features will be limited.")

from ...core.types import Result, create_success, create_failure
from ...core.utils import generate_id, current_timestamp, safe_get
from ...core.errors import ApplicationError as LearningError, ErrorSeverity
from ...core.utils.portable_storage import get_storage_manager
from .types import (
    LearningPoint, LearningPattern, LearningStatus, GeneratedTactic,
    GeneratedHeuristic, GeneratedKnowledge, LearningCategory, PatternType,
    KoreanAnalysisResult, LearningMetrics, DatabaseInterface,
    ConversationMemoryInterface
)

logger = logging.getLogger(__name__)


class AutoLearningSystem:
    """
    자동 학습 시스템

    Features:
    - 한국어 대화 패턴 자동 감지
    - 성공/실패 패턴 학습
    - 사용자 선호도 추출
    - 자동 전술/휴리스틱 생성
    - KoNLPy 기반 한국어 NLP 최적화
    """

    def __init__(
        self,
        database: DatabaseInterface,
        conversation_memory: ConversationMemoryInterface,
        storage_path: Optional[str] = None,
        enable_korean_nlp: bool = True
    ):
        self.database = database
        self.conversation_memory = conversation_memory
        # 포터블 저장소 사용
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            storage_manager = get_storage_manager()
            self.storage_path = storage_manager.get_memory_storage_path("learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 학습 데이터
        self.learning_points: List[LearningPoint] = []
        self.generated_tactics: List[GeneratedTactic] = []
        self.generated_heuristics: List[GeneratedHeuristic] = []
        self.metrics = LearningMetrics()

        # 한국어 NLP 설정
        self.enable_korean_nlp = enable_korean_nlp and KONLPY_AVAILABLE
        if self.enable_korean_nlp:
            self.korean_analyzer = Okt()
            logger.info("Korean NLP analyzer initialized with KoNLPy")
        else:
            self.korean_analyzer = None
            logger.warning("Korean NLP disabled or KoNLPy not available")

        # 학습 패턴 정의 (한국어 최적화)
        self.learning_patterns = self._initialize_korean_patterns()

        # 저장 동기화 락 (첫 비동기 저장 시점에 생성)

        self._save_lock: Optional[asyncio.Lock] = None

        # 데이터 로드
        self._load_learning_data()

    def _initialize_korean_patterns(self) -> List[LearningPattern]:
        """한국어 최적화된 학습 패턴 초기화"""
        return [
            # 성공 패턴 (더 다양한 한국어 표현)
            LearningPattern(
                pattern_type=PatternType.SUCCESS,
                keywords=[
                    "해결됐어", "완료", "성공", "좋아", "맞아", "정확해", "도움이 됐어",
                    "감사", "잘 작동해", "문제없어", "됐다", "성공했어", "해결했어",
                    "훌륭해", "완벽해", "좋네", "잘 되네", "OK", "오케이"
                ],
                context_indicators=[
                    "문제", "오류", "버그", "수정", "개선", "해결", "처리", "구현",
                    "작업", "태스크", "이슈", "에러", "실행"
                ],
                confidence_threshold=0.8,
                extraction_rule="Extract the method or approach that led to success",
                language="ko"
            ),

            # 실패 패턴
            LearningPattern(
                pattern_type=PatternType.FAILURE,
                keywords=[
                    "안돼", "실패", "오류", "문제", "작동하지 않아", "안 되네", "에러",
                    "안 돼", "안돼네", "실패했어", "문제야", "이상해", "버그",
                    "작동 안해", "안 됨", "안됨", "불가능"
                ],
                context_indicators=[
                    "시도", "실행", "테스트", "적용", "사용", "작업", "처리",
                    "구현", "설치", "설정", "실행"
                ],
                confidence_threshold=0.7,
                extraction_rule="Extract the approach that should be avoided",
                language="ko"
            ),

            # 선호도 패턴
            LearningPattern(
                pattern_type=PatternType.PREFERENCE,
                keywords=[
                    "선호해", "좋아해", "싫어해", "원해", "필요해", "중요해",
                    "마음에 들어", "별로야", "그냥", "더 나아", "선호",
                    "좋아", "싫어", "원함", "필요함"
                ],
                context_indicators=[
                    "방법", "방식", "도구", "언어", "프레임워크", "라이브러리",
                    "기술", "접근법", "스타일", "패턴"
                ],
                confidence_threshold=0.6,
                extraction_rule="Extract user preferences for future reference",
                language="ko"
            ),

            # 지식 갭 패턴
            LearningPattern(
                pattern_type=PatternType.KNOWLEDGE,
                keywords=[
                    "몰랐어", "처음 알았어", "배웠어", "이해했어", "신기해",
                    "몰랐네", "처음", "새로워", "배움", "이해", "알게 됐어",
                    "깨달았어", "알았어"
                ],
                context_indicators=[
                    "개념", "원리", "방법", "기술", "이론", "방식", "시스템",
                    "구조", "패턴", "알고리즘"
                ],
                confidence_threshold=0.7,
                extraction_rule="Extract new knowledge for teaching opportunities",
                language="ko"
            ),

            # 성능 패턴
            LearningPattern(
                pattern_type=PatternType.PERFORMANCE,
                keywords=[
                    "느려", "빨라", "성능", "최적화", "빠르게", "느리게",
                    "속도", "효율", "개선", "향상"
                ],
                context_indicators=[
                    "실행", "처리", "응답", "로딩", "렌더링", "계산",
                    "쿼리", "네트워크", "메모리"
                ],
                confidence_threshold=0.7,
                extraction_rule="Extract performance insights",
                language="ko"
            )
        ]

    async def analyze_learning_opportunities(
        self,
        user_message: str,
        paca_response: str,
        conversation_id: Optional[str] = None
    ) -> Result[List[LearningPoint]]:
        """대화에서 자동으로 학습 포인트 감지 및 분석"""
        try:
            detected_patterns = await self._detect_learning_patterns(user_message, paca_response)
            learning_points = []

            for pattern in detected_patterns:
                learning_point = await self._extract_learning_point(
                    user_message, paca_response, conversation_id, pattern
                )

                if learning_point and learning_point.confidence >= pattern.confidence_threshold:
                    self.learning_points.append(learning_point)
                    learning_points.append(learning_point)

                    # 메트릭 업데이트
                    self.metrics.update_metrics(learning_point)

                    # 즉시 지식 생성 시도
                    await self._attempt_knowledge_generation(learning_point)

                    logger.info(f"New learning point created: {learning_point.category.value}")

            # 데이터 저장
            await self._save_learning_data()

            return create_success(learning_points)

        except Exception as e:
            error_msg = f"Failed to analyze learning opportunities: {str(e)}"
            logger.error(error_msg)
            return create_failure(error_msg)

    def get_learning_status(self) -> LearningStatus:
        """자동 학습 상태 조회"""
        recent_threshold = time.time() - (24 * 60 * 60)  # 24시간 전
        recent_learning = len([
            lp for lp in self.learning_points
            if lp.created_at > recent_threshold
        ])

        return LearningStatus(
            learning_points=len(self.learning_points),
            generated_tactics=len(self.generated_tactics),
            generated_heuristics=len(self.generated_heuristics),
            recent_learning=recent_learning,
            total_conversations_analyzed=self.metrics.total_learning_points,
            average_confidence=self.metrics.average_confidence,
            last_learning_at=max([lp.created_at for lp in self.learning_points], default=None),
            active_patterns=len(self.learning_patterns)
        )

    def get_generated_knowledge(self) -> GeneratedKnowledge:
        """생성된 지식 조회"""
        tactics_data = []
        for tactic in self.generated_tactics:
            tactics_data.append({
                "name": tactic.name,
                "description": tactic.description,
                "effectiveness": tactic.effectiveness,
                "success_rate": tactic.success_rate,
                "applications": tactic.total_applications,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tactic.created_at))
            })

        heuristics_data = []
        for heuristic in self.generated_heuristics:
            heuristics_data.append({
                "pattern": heuristic.pattern,
                "avoidance_rule": heuristic.avoidance_rule,
                "effectiveness": heuristic.effectiveness,
                "avoidance_rate": heuristic.avoidance_rate,
                "triggers": heuristic.triggered_count,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(heuristic.created_at))
            })

        effectiveness_summary = {
            "tactics_avg": sum(t.effectiveness for t in self.generated_tactics) / max(len(self.generated_tactics), 1),
            "heuristics_avg": sum(h.effectiveness for h in self.generated_heuristics) / max(len(self.generated_heuristics), 1),
            "overall_learning_rate": self.metrics.learning_rate
        }

        return GeneratedKnowledge(
            tactics=tactics_data,
            heuristics=heuristics_data,
            effectiveness_summary=effectiveness_summary
        )

    async def _detect_learning_patterns(
        self,
        user_message: str,
        paca_response: str
    ) -> List[LearningPattern]:
        """학습 패턴 감지 (한국어 NLP 최적화)"""
        combined_text = f"{user_message} {paca_response}".lower()
        detected_patterns = []

        # 한국어 NLP 분석
        if self.enable_korean_nlp:
            korean_analysis = await self._analyze_korean_text(combined_text)
        else:
            korean_analysis = None

        for pattern in self.learning_patterns:
            # 키워드 매칭 점수
            keyword_score = self._calculate_keyword_score(combined_text, pattern.keywords)

            # 컨텍스트 매칭 점수
            context_score = self._calculate_keyword_score(combined_text, pattern.context_indicators)

            # 한국어 NLP 보조 점수
            nlp_score = 0.0
            if korean_analysis:
                nlp_score = self._calculate_nlp_score(korean_analysis, pattern)

            # 종합 점수 계산
            total_score = (keyword_score * 0.4) + (context_score * 0.4) + (nlp_score * 0.2)

            if total_score >= pattern.confidence_threshold * 0.7:  # 약간 낮은 임계값으로 후보 선별
                detected_patterns.append(pattern)

        return detected_patterns

    async def _analyze_korean_text(self, text: str) -> Optional[KoreanAnalysisResult]:
        """한국어 텍스트 분석"""
        if not self.korean_analyzer:
            return None

        try:
            # 형태소 분석
            morphs = self.korean_analyzer.morphs(text)
            pos_tags = self.korean_analyzer.pos(text)
            nouns = self.korean_analyzer.nouns(text)

            # 품사별 분류
            verbs = [word for word, pos in pos_tags if pos.startswith('V')]
            adjectives = [word for word, pos in pos_tags if pos.startswith('A')]

            # 간단한 감정 분석 (긍정/부정 키워드 기반)
            positive_words = {"좋", "훌륭", "완벽", "성공", "해결", "감사", "만족"}
            negative_words = {"나쁘", "실패", "문제", "오류", "에러", "안됨", "불가능"}

            positive_count = sum(1 for word in morphs if any(pos in word for pos in positive_words))
            negative_count = sum(1 for word in morphs if any(neg in word for neg in negative_words))

            sentiment_score = 0.0
            if positive_count + negative_count > 0:
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)

            # 분석 신뢰도 계산
            confidence = min(1.0, len(morphs) / 50.0)  # 단어 수가 많을수록 신뢰도 증가

            return KoreanAnalysisResult(
                morphs=morphs,
                pos_tags=pos_tags,
                nouns=nouns,
                verbs=verbs,
                adjectives=adjectives,
                sentiment_score=sentiment_score,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Korean text analysis failed: {str(e)}")
            return None

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        matched_count = 0
        for keyword in keywords:
            if keyword.lower() in text:
                matched_count += 1

        return matched_count / max(len(keywords), 1)

    def _calculate_nlp_score(self, analysis: KoreanAnalysisResult, pattern: LearningPattern) -> float:
        """NLP 분석 기반 점수 계산"""
        score = 0.0

        # 감정 점수 반영
        if pattern.pattern_type == PatternType.SUCCESS and analysis.positive_sentiment:
            score += 0.3
        elif pattern.pattern_type == PatternType.FAILURE and analysis.negative_sentiment:
            score += 0.3

        # 품사 분석 반영
        if pattern.pattern_type == PatternType.KNOWLEDGE and len(analysis.nouns) > 3:
            score += 0.2

        if pattern.pattern_type == PatternType.PREFERENCE and len(analysis.adjectives) > 1:
            score += 0.2

        # 분석 신뢰도 반영
        score *= analysis.confidence

        return min(score, 1.0)

    async def _extract_learning_point(
        self,
        user_message: str,
        paca_response: str,
        conversation_id: Optional[str],
        pattern: LearningPattern
    ) -> Optional[LearningPoint]:
        """학습 포인트 추출"""
        try:
            # 카테고리 매핑
            category_map = {
                PatternType.SUCCESS: LearningCategory.SUCCESS_PATTERN,
                PatternType.FAILURE: LearningCategory.ERROR_PATTERN,
                PatternType.PREFERENCE: LearningCategory.USER_PREFERENCE,
                PatternType.KNOWLEDGE: LearningCategory.KNOWLEDGE_GAP,
                PatternType.PERFORMANCE: LearningCategory.PERFORMANCE_ISSUE
            }

            category = category_map.get(pattern.pattern_type, LearningCategory.SUCCESS_PATTERN)

            # 지식 추출
            extracted_knowledge = await self._extract_knowledge_by_type(
                user_message, paca_response, pattern.pattern_type
            )

            if not extracted_knowledge or len(extracted_knowledge.strip()) < 10:
                return None

            # 컨텍스트 추출
            context = self._extract_context(user_message, paca_response)

            # 신뢰도 계산
            confidence = self._calculate_confidence(user_message, paca_response, pattern)

            return LearningPoint(
                user_message=user_message,
                paca_response=paca_response,
                context=context,
                category=category,
                confidence=confidence,
                extracted_knowledge=extracted_knowledge,
                conversation_id=conversation_id,
                source_pattern=pattern.pattern_type.value
            )

        except Exception as e:
            logger.error(f"Failed to extract learning point: {str(e)}")
            return None

    async def _extract_knowledge_by_type(
        self,
        user_message: str,
        paca_response: str,
        pattern_type: PatternType
    ) -> str:
        """패턴 타입별 지식 추출"""
        if pattern_type == PatternType.SUCCESS:
            return self._extract_success_method(user_message, paca_response)
        elif pattern_type == PatternType.FAILURE:
            return self._extract_failure_pattern(user_message, paca_response)
        elif pattern_type == PatternType.PREFERENCE:
            return self._extract_user_preference(user_message, paca_response)
        elif pattern_type == PatternType.KNOWLEDGE:
            return self._extract_new_knowledge(user_message, paca_response)
        elif pattern_type == PatternType.PERFORMANCE:
            return self._extract_performance_insight(user_message, paca_response)
        else:
            return ""

    def _extract_success_method(self, user_message: str, paca_response: str) -> str:
        """성공 방법 추출"""
        success_keywords = ["해결", "수정", "완료", "성공", "작동", "고침", "처리"]
        lines = paca_response.split('\n')
        relevant_lines = []

        for line in lines:
            if any(keyword in line for keyword in success_keywords):
                relevant_lines.append(line.strip())

        result = ' '.join(relevant_lines)
        return result[:300] if result else user_message[:200]

    def _extract_failure_pattern(self, user_message: str, paca_response: str) -> str:
        """실패 패턴 추출"""
        error_keywords = ["오류", "에러", "실패", "문제", "안됨", "작동하지 않음", "버그"]
        combined_text = f"{user_message} {paca_response}"
        lines = combined_text.split('\n')
        relevant_lines = []

        for line in lines:
            if any(keyword in line for keyword in error_keywords):
                relevant_lines.append(line.strip())

        result = ' '.join(relevant_lines)
        return result[:300] if result else user_message[:200]

    def _extract_user_preference(self, user_message: str, paca_response: str) -> str:
        """사용자 선호도 추출"""
        preference_keywords = ["선호", "좋아", "싫어", "원해", "필요", "중요", "마음에"]
        lines = user_message.split('\n')
        relevant_lines = []

        for line in lines:
            if any(keyword in line for keyword in preference_keywords):
                relevant_lines.append(line.strip())

        result = ' '.join(relevant_lines)
        return result[:300] if result else user_message[:200]

    def _extract_new_knowledge(self, user_message: str, paca_response: str) -> str:
        """새로운 지식 추출"""
        knowledge_keywords = ["몰랐", "처음", "배웠", "이해", "신기", "알았", "깨달았"]
        combined_text = f"{user_message} {paca_response}"
        lines = combined_text.split('\n')
        relevant_lines = []

        for line in lines:
            if any(keyword in line for keyword in knowledge_keywords):
                relevant_lines.append(line.strip())

        result = ' '.join(relevant_lines)
        return result[:300] if result else combined_text[:200]

    def _extract_performance_insight(self, user_message: str, paca_response: str) -> str:
        """성능 인사이트 추출"""
        performance_keywords = ["느려", "빨라", "성능", "최적화", "속도", "효율"]
        combined_text = f"{user_message} {paca_response}"
        lines = combined_text.split('\n')
        relevant_lines = []

        for line in lines:
            if any(keyword in line for keyword in performance_keywords):
                relevant_lines.append(line.strip())

        result = ' '.join(relevant_lines)
        return result[:300] if result else combined_text[:200]

    def _extract_context(self, user_message: str, paca_response: str) -> str:
        """컨텍스트 추출"""
        context_keywords = {
            "프로그래밍": "programming",
            "개발": "development",
            "UI": "ui",
            "API": "api",
            "데이터베이스": "database",
            "설계": "design",
            "디버그": "debug",
            "웹": "web",
            "앱": "app",
            "시스템": "system"
        }

        combined_text = f"{user_message} {paca_response}".lower()

        for keyword, context in context_keywords.items():
            if keyword.lower() in combined_text:
                return context

        return "general"

    def _calculate_confidence(
        self,
        user_message: str,
        paca_response: str,
        pattern: LearningPattern
    ) -> float:
        """신뢰도 계산"""
        base_confidence = 0.5
        combined_text = f"{user_message} {paca_response}".lower()

        # 키워드 매칭 점수
        keyword_matches = sum(
            1 for keyword in pattern.keywords
            if keyword.lower() in combined_text
        )
        base_confidence += keyword_matches * 0.1

        # 컨텍스트 매칭 점수
        context_matches = sum(
            1 for indicator in pattern.context_indicators
            if indicator.lower() in combined_text
        )
        base_confidence += context_matches * 0.15

        # 메시지 길이 보너스
        if len(combined_text) > 100:
            base_confidence += 0.1

        # 한국어 NLP 보너스
        if self.enable_korean_nlp:
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    async def _attempt_knowledge_generation(self, learning_point: LearningPoint) -> None:
        """지식 생성 시도"""
        try:
            if learning_point.category == LearningCategory.SUCCESS_PATTERN:
                await self._generate_tactic(learning_point)
            elif learning_point.category == LearningCategory.ERROR_PATTERN:
                await self._generate_heuristic(learning_point)
        except Exception as e:
            logger.error(f"Knowledge generation failed: {str(e)}")

    async def _generate_tactic(self, learning_point: LearningPoint) -> None:
        """전술 생성"""
        tactic = GeneratedTactic(
            name=f"자동생성: {learning_point.context} 해결 전술",
            description=learning_point.extracted_knowledge,
            context=learning_point.context,
            source_conversations=[learning_point.id],
            tags=[learning_point.context, "auto_generated", "success"]
        )

        # 유사한 전술 검사
        similar_tactic = self._find_similar_tactic(tactic)
        if not similar_tactic:
            self.generated_tactics.append(tactic)
            self.database.add_experience(tactic.name, tactic.description)
            logger.info(f"New tactic generated: {tactic.name}")
        else:
            # 기존 전술 강화
            similar_tactic.source_conversations.append(learning_point.id)
            similar_tactic.apply(success=True)

    async def _generate_heuristic(self, learning_point: LearningPoint) -> None:
        """휴리스틱 생성"""
        pattern = self._extract_pattern(learning_point.extracted_knowledge)

        heuristic = GeneratedHeuristic(
            pattern=pattern,
            avoidance_rule=f"{learning_point.context}에서 {learning_point.extracted_knowledge} 상황을 피할 것",
            context=learning_point.context,
            source_conversations=[learning_point.id],
            severity="medium"
        )

        # 유사한 휴리스틱 검사
        similar_heuristic = self._find_similar_heuristic(heuristic)
        if not similar_heuristic:
            self.generated_heuristics.append(heuristic)
            self.database.add_heuristic(heuristic.avoidance_rule)
            logger.info(f"New heuristic generated: {heuristic.pattern}")
        else:
            # 기존 휴리스틱 강화
            similar_heuristic.source_conversations.append(learning_point.id)
            similar_heuristic.trigger(avoided=True)

    def _find_similar_tactic(self, tactic: GeneratedTactic) -> Optional[GeneratedTactic]:
        """유사한 전술 찾기"""
        for existing_tactic in self.generated_tactics:
            if self._calculate_text_similarity(existing_tactic.description, tactic.description) > 0.7:
                return existing_tactic
        return None

    def _find_similar_heuristic(self, heuristic: GeneratedHeuristic) -> Optional[GeneratedHeuristic]:
        """유사한 휴리스틱 찾기"""
        for existing_heuristic in self.generated_heuristics:
            if self._calculate_text_similarity(existing_heuristic.pattern, heuristic.pattern) > 0.7:
                return existing_heuristic
        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (Jaccard 유사도)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _extract_pattern(self, description: str) -> str:
        """패턴 추출"""
        words = description.split()
        # 의미있는 단어들만 선택 (길이 3 이상)
        meaningful_words = [word for word in words if len(word) >= 3]
        return ' '.join(meaningful_words[:5])

    async def _save_learning_data(self) -> None:
        """학습 데이터 저장"""
        if self._save_lock is None:
            self._save_lock = asyncio.Lock()

        async with self._save_lock:

            try:
                learning_points_file = self.storage_path / "learning_points.json"
                learning_points_data = []
                for lp in self.learning_points:
                    lp_snapshot = copy.deepcopy(lp.__dict__)
                    lp_snapshot["category"] = lp.category.value
                    lp_snapshot["created_at"] = lp.created_at
                    lp_snapshot["updated_at"] = lp.updated_at
                    learning_points_data.append(lp_snapshot)

                tactics_file = self.storage_path / "generated_tactics.json"
                tactics_data = [copy.deepcopy(tactic.__dict__) for tactic in self.generated_tactics]

                heuristics_file = self.storage_path / "generated_heuristics.json"
                heuristics_data = [copy.deepcopy(heuristic.__dict__) for heuristic in self.generated_heuristics]

                metrics_file = self.storage_path / "learning_metrics.json"
                metrics_data = copy.deepcopy(self.metrics.__dict__)

                artifacts: List[Tuple[Path, Any]] = [
                    (learning_points_file, learning_points_data),
                    (tactics_file, tactics_data),
                    (heuristics_file, heuristics_data),
                    (metrics_file, metrics_data),
                ]

                for path, data in artifacts:
                    await self._write_json_artifact(path, data)

            except Exception as e:
                logger.error(f"Failed to save learning data: {str(e)}")

    async def _write_json_artifact(self, path: Path, data: Any) -> None:
        """비동기적으로 JSON 아티팩트를 저장"""
        await asyncio.to_thread(self._write_json_file, path, data)

    @staticmethod
    def _write_json_file(path: Path, data: Any) -> None:
        """JSON 데이터를 파일로 저장"""
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding='utf-8')

    def _load_learning_data(self) -> None:
        """학습 데이터 로드"""
        try:
            loaded_learning_points: List[LearningPoint] = []
            loaded_tactics: List[GeneratedTactic] = []
            loaded_heuristics: List[GeneratedHeuristic] = []
            loaded_metrics = self.metrics

            # 학습 포인트 로드
            learning_points_file = self.storage_path / "learning_points.json"
            if learning_points_file.exists():
                with open(learning_points_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                lp_fields = {field.name for field in fields(LearningPoint)}
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    item_data = dict(item)
                    if 'category' in item_data:
                        try:
                            item_data['category'] = LearningCategory(item_data['category'])
                        except ValueError:
                            logger.warning("Unknown learning category encountered during load: %s", item_data['category'])
                            continue

                    filtered_item = {k: item_data.get(k) for k in lp_fields if k in item_data}
                    try:
                        loaded_learning_points.append(LearningPoint(**filtered_item))
                    except (TypeError, ValueError) as e:
                        logger.warning("Skipping invalid learning point entry: %s", e)

            # 전술 로드
            tactics_file = self.storage_path / "generated_tactics.json"
            if tactics_file.exists():
                with open(tactics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tactic_fields = {field.name for field in fields(GeneratedTactic)}
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    filtered_item = {k: item.get(k) for k in tactic_fields if k in item}
                    try:
                        loaded_tactics.append(GeneratedTactic(**filtered_item))
                    except (TypeError, ValueError) as e:
                        logger.warning("Skipping invalid generated tactic entry: %s", e)

            # 휴리스틱 로드
            heuristics_file = self.storage_path / "generated_heuristics.json"
            if heuristics_file.exists():
                with open(heuristics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                heuristic_fields = {field.name for field in fields(GeneratedHeuristic)}
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    filtered_item = {k: item.get(k) for k in heuristic_fields if k in item}
                    try:
                        loaded_heuristics.append(GeneratedHeuristic(**filtered_item))
                    except (TypeError, ValueError) as e:
                        logger.warning("Skipping invalid generated heuristic entry: %s", e)

            # 메트릭 로드
            metrics_file = self.storage_path / "learning_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metrics_fields = {field.name for field in fields(LearningMetrics)}
                filtered_metrics = {k: data.get(k) for k in metrics_fields if k in data}
                try:
                    loaded_metrics = LearningMetrics(**filtered_metrics)
                except (TypeError, ValueError) as e:
                    logger.warning("Failed to load learning metrics, using defaults: %s", e)
                    loaded_metrics = LearningMetrics()

            # 상태 반영
            self.learning_points = loaded_learning_points
            self.generated_tactics = loaded_tactics
            self.generated_heuristics = loaded_heuristics
            self.metrics = loaded_metrics

            # 오래된 데이터 정리 (상태 복원 후 실행)
            self._cleanup_old_learning_data()

        except Exception as e:
            logger.error(f"Failed to load learning data: {str(e)}")

    def _cleanup_old_learning_data(self) -> None:
        """오래된 학습 데이터 정리"""
        try:
            current_time = time.time()
            ninety_days_ago = current_time - (90 * 24 * 60 * 60)
            thirty_days_ago = current_time - (30 * 24 * 60 * 60)

            # 90일 이상 된 학습 포인트 제거
            self.learning_points = [
                lp for lp in self.learning_points
                if lp.created_at > ninety_days_ago
            ]

            # 효과성이 낮거나 사용되지 않는 전술/휴리스틱 제거
            self.generated_tactics = [
                t for t in self.generated_tactics
                if t.effectiveness >= 0.3 or (t.last_used and t.last_used > thirty_days_ago)
            ]

            self.generated_heuristics = [
                h for h in self.generated_heuristics
                if h.effectiveness >= 0.3 or (h.last_triggered and h.last_triggered > thirty_days_ago)
            ]

        except Exception as e:
            logger.error(f"Failed to cleanup old learning data: {str(e)}")