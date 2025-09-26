"""
Sentiment Analysis System
PACA v5 감정 분석 시스템 - 한국어 텍스트 감정 분석 및 감정 추적
"""

import re
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ..core.types.base import ID, Timestamp, create_id, current_timestamp
from ..core.utils.logger import create_logger
from ..core.errors.base import PacaError

class EmotionType(Enum):
    """감정 타입"""
    HAPPY = 'happy'           # 기쁨
    SAD = 'sad'               # 슬픔
    ANGRY = 'angry'           # 분노
    FEAR = 'fear'             # 두려움
    SURPRISE = 'surprise'     # 놀람
    DISGUST = 'disgust'       # 혐오
    NEUTRAL = 'neutral'       # 중립
    FRUSTRATED = 'frustrated' # 답답함
    EXCITED = 'excited'       # 흥분
    CALM = 'calm'            # 평온

@dataclass
class SentimentScore:
    """감정 점수"""
    positive: float = 0.0      # 긍정 점수 (0.0 ~ 1.0)
    negative: float = 0.0      # 부정 점수 (0.0 ~ 1.0)
    neutral: float = 1.0       # 중립 점수 (0.0 ~ 1.0)
    confidence: float = 0.0    # 신뢰도 (0.0 ~ 1.0)

    @property
    def dominant_sentiment(self) -> str:
        """주요 감정 반환"""
        scores = {
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral
        }
        return max(scores, key=scores.get)

@dataclass
class SentimentResult:
    """감정 분석 결과"""
    text: str
    emotion_type: EmotionType
    sentiment_score: SentimentScore
    detected_keywords: List[str] = field(default_factory=list)
    emotion_intensity: float = 0.0  # 감정 강도 (0.0 ~ 1.0)
    context_clues: List[str] = field(default_factory=list)
    timestamp: Timestamp = field(default_factory=current_timestamp)
    analysis_id: ID = field(default_factory=create_id)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'text': self.text,
            'emotion_type': self.emotion_type.value,
            'sentiment_score': {
                'positive': self.sentiment_score.positive,
                'negative': self.sentiment_score.negative,
                'neutral': self.sentiment_score.neutral,
                'confidence': self.sentiment_score.confidence,
                'dominant': self.sentiment_score.dominant_sentiment
            },
            'detected_keywords': self.detected_keywords,
            'emotion_intensity': self.emotion_intensity,
            'context_clues': self.context_clues,
            'timestamp': self.timestamp,
            'analysis_id': self.analysis_id
        }

@dataclass
class EmotionAnalysisResult:
    """감정 분석 상세 결과"""
    primary_emotion: EmotionType
    secondary_emotions: List[Tuple[EmotionType, float]]
    emotion_progression: List[EmotionType]
    emotional_markers: Dict[str, List[str]]
    linguistic_features: Dict[str, Any]

@dataclass
class SentimentTrend:
    """감정 추세"""
    user_id: str
    time_window: timedelta
    emotion_history: List[SentimentResult]
    dominant_emotions: List[EmotionType]
    trend_direction: str  # 'improving', 'declining', 'stable'
    average_intensity: float

@dataclass
class TextAnalysisConfig:
    """텍스트 분석 설정"""
    language: str = 'ko'
    enable_context_analysis: bool = True
    enable_keyword_extraction: bool = True
    enable_emotion_tracking: bool = True
    min_text_length: int = 1
    max_text_length: int = 10000

class SentimentAnalyzer:
    """감정 분석기"""

    def __init__(self, config: Optional[TextAnalysisConfig] = None):
        self.config = config or TextAnalysisConfig()
        self.logger = create_logger(__name__)

        # 감정 키워드 사전 (한국어)
        self.emotion_keywords = self._load_emotion_keywords()

        # 감정 패턴 정규식
        self.emotion_patterns = self._compile_emotion_patterns()

        # 감정 분석 히스토리 (사용자별)
        self.emotion_history: Dict[str, List[SentimentResult]] = {}

        # 문맥 분석기
        self.context_analyzer = ContextAnalyzer()

        self.logger.info("감정 분석기 초기화 완료")

    def _load_emotion_keywords(self) -> Dict[EmotionType, List[str]]:
        """감정 키워드 사전 로드"""
        return {
            EmotionType.HAPPY: [
                '기쁘다', '행복하다', '즐겁다', '좋다', '멋지다', '훌륭하다',
                '최고다', '완벽하다', '사랑한다', '고맙다', '감사하다',
                '웃다', '미소', '기뻐하다', '만족하다', '성공하다',
                '축하', '기념', '즐거운', '유쾌한', '상쾌한'
            ],
            EmotionType.SAD: [
                '슬프다', '우울하다', '속상하다', '아프다', '힘들다',
                '외롭다', '쓸쓸하다', '눈물', '울다', '슬퍼하다',
                '절망하다', '후회하다', '안타깝다', '가슴아프다',
                '비참하다', '암울하다', '침울하다', '어둡다'
            ],
            EmotionType.ANGRY: [
                '화나다', '짜증나다', '분노하다', '열받다', '빡치다',
                '억울하다', '불만이다', '불쾌하다', '성나다', '약오르다',
                '격분하다', '분개하다', '노하다', '분하다', '원망하다',
                '증오하다', '미워하다', '욱하다', '뚜껑열린다'
            ],
            EmotionType.FEAR: [
                '무섭다', '두렵다', '불안하다', '걱정되다', '떨다',
                '공포', '무서워하다', '겁나다', '조마조마하다',
                '심장이 뛰다', '오싹하다', '섬뜩하다', '무시무시하다',
                '간담이 서늘하다', '간이 콩알만하다'
            ],
            EmotionType.SURPRISE: [
                '놀라다', '깜짝', '어머', '헉', '와', '오',
                '예상외다', '뜻밖이다', '의외다', '당황하다',
                '어리둥절하다', '황당하다', '멍하다', '어안이 벙벙하다'
            ],
            EmotionType.FRUSTRATED: [
                '답답하다', '막막하다', '갑갑하다', '곤란하다',
                '어렵다', '복잡하다', '혼란스럽다', '난감하다',
                '스트레스', '부담스럽다', '귀찮다', '피곤하다'
            ],
            EmotionType.EXCITED: [
                '신나다', '흥미롭다', '재미있다', '설레다',
                '기대되다', '두근두근', '들뜨다', '활기차다',
                '생기있다', '활력있다', '역동적이다'
            ],
            EmotionType.CALM: [
                '평온하다', '차분하다', '조용하다', '안정되다',
                '편안하다', '평화롭다', '고요하다', '여유롭다',
                '느긋하다', '천천히', '조심스럽다'
            ]
        }

    def _compile_emotion_patterns(self) -> Dict[EmotionType, List[re.Pattern]]:
        """감정 패턴 정규식 컴파일"""
        patterns = {}

        for emotion_type, keywords in self.emotion_keywords.items():
            emotion_patterns = []
            for keyword in keywords:
                # 기본 패턴
                pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                emotion_patterns.append(pattern)

                # 강조 패턴 (반복, 느낌표 등)
                emphasis_pattern = re.compile(rf'{re.escape(keyword)}[!?]*', re.IGNORECASE)
                emotion_patterns.append(emphasis_pattern)

            patterns[emotion_type] = emotion_patterns

        return patterns

    async def analyze(self, text: str, user_id: Optional[str] = None) -> SentimentResult:
        """텍스트 감정 분석"""
        if not text or len(text) < self.config.min_text_length:
            return self._create_neutral_result(text)

        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        try:
            # 1. 전처리
            processed_text = self._preprocess_text(text)

            # 2. 감정 키워드 매칭
            emotion_scores = self._calculate_emotion_scores(processed_text)

            # 3. 주요 감정 결정
            primary_emotion = self._determine_primary_emotion(emotion_scores)

            # 4. 감정 강도 계산
            emotion_intensity = self._calculate_emotion_intensity(emotion_scores, processed_text)

            # 5. 감정 점수 계산
            sentiment_score = self._calculate_sentiment_score(emotion_scores)

            # 6. 키워드 추출
            detected_keywords = self._extract_keywords(processed_text, primary_emotion)

            # 7. 문맥 단서 추출
            context_clues = self._extract_context_clues(processed_text)

            # 8. 결과 생성
            result = SentimentResult(
                text=text,
                emotion_type=primary_emotion,
                sentiment_score=sentiment_score,
                detected_keywords=detected_keywords,
                emotion_intensity=emotion_intensity,
                context_clues=context_clues
            )

            # 9. 히스토리 저장 (사용자 ID가 있는 경우)
            if user_id and self.config.enable_emotion_tracking:
                self._save_emotion_history(user_id, result)

            self.logger.debug(f"감정 분석 완료: {primary_emotion.value} (강도: {emotion_intensity:.2f})")

            return result

        except Exception as e:
            self.logger.error(f"감정 분석 실패: {e}")
            return self._create_neutral_result(text, error=str(e))

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본 정리
        processed = text.strip()

        # 특수문자 정규화
        processed = re.sub(r'[ㅋㅎ]+', ' 웃음 ', processed)  # ㅋㅋㅋ, ㅎㅎㅎ
        processed = re.sub(r'[ㅠㅜ]+', ' 슬픔 ', processed)  # ㅠㅠㅠ, ㅜㅜㅜ
        processed = re.sub(r'\.{3,}', '...', processed)      # 말줄임표 정규화
        processed = re.sub(r'!{2,}', '!!', processed)        # 느낌표 정규화
        processed = re.sub(r'\?{2,}', '??', processed)       # 물음표 정규화

        # 이모티콘 처리
        emoticon_patterns = {
            r'[:;]-?[\)\(]': ' 웃음 ',
            r'[:;]-?[pP]': ' 웃음 ',
            r'[:;]-?[\[\]]': ' 슬픔 ',
            r'T[_\.]T': ' 슬픔 ',
            r'[><]': ' 화남 '
        }

        for pattern, replacement in emoticon_patterns.items():
            processed = re.sub(pattern, replacement, processed)

        return processed

    def _calculate_emotion_scores(self, text: str) -> Dict[EmotionType, float]:
        """감정 점수 계산"""
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}

        for emotion_type, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # 매치 수와 텍스트 길이를 고려한 점수
                    score = len(matches) / max(len(text.split()), 1)
                    emotion_scores[emotion_type] += score

        return emotion_scores

    def _determine_primary_emotion(self, emotion_scores: Dict[EmotionType, float]) -> EmotionType:
        """주요 감정 결정"""
        if not any(emotion_scores.values()):
            return EmotionType.NEUTRAL

        # 최고 점수 감정 반환
        return max(emotion_scores, key=emotion_scores.get)

    def _calculate_emotion_intensity(self, emotion_scores: Dict[EmotionType, float], text: str) -> float:
        """감정 강도 계산"""
        max_score = max(emotion_scores.values()) if emotion_scores.values() else 0.0

        # 강조 표현 가중치
        intensity_modifiers = 0.0

        # 느낌표, 물음표
        intensity_modifiers += text.count('!') * 0.1
        intensity_modifiers += text.count('?') * 0.05

        # 대문자 (한국어에서는 제한적)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        intensity_modifiers += uppercase_ratio * 0.2

        # 반복 표현
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        intensity_modifiers += repeated_chars * 0.05

        # 최종 강도 계산 (0.0 ~ 1.0)
        intensity = min(max_score + intensity_modifiers, 1.0)

        return intensity

    def _calculate_sentiment_score(self, emotion_scores: Dict[EmotionType, float]) -> SentimentScore:
        """감정 점수 계산"""
        positive_emotions = [EmotionType.HAPPY, EmotionType.EXCITED, EmotionType.SURPRISE]
        negative_emotions = [EmotionType.SAD, EmotionType.ANGRY, EmotionType.FEAR, EmotionType.FRUSTRATED]
        neutral_emotions = [EmotionType.NEUTRAL, EmotionType.CALM]

        positive_score = sum(emotion_scores.get(emotion, 0.0) for emotion in positive_emotions)
        negative_score = sum(emotion_scores.get(emotion, 0.0) for emotion in negative_emotions)
        neutral_score = sum(emotion_scores.get(emotion, 0.0) for emotion in neutral_emotions)

        # 정규화
        total = positive_score + negative_score + neutral_score
        if total == 0:
            return SentimentScore(neutral=1.0, confidence=0.5)

        positive_norm = positive_score / total
        negative_norm = negative_score / total
        neutral_norm = neutral_score / total

        # 신뢰도 계산 (점수 분산이 클수록 높은 신뢰도)
        scores = [positive_norm, negative_norm, neutral_norm]
        max_score = max(scores)
        confidence = max_score * 2 - 1 if max_score > 0.5 else 0.0

        return SentimentScore(
            positive=positive_norm,
            negative=negative_norm,
            neutral=neutral_norm,
            confidence=min(confidence, 1.0)
        )

    def _extract_keywords(self, text: str, emotion_type: EmotionType) -> List[str]:
        """키워드 추출"""
        if not self.config.enable_keyword_extraction:
            return []

        keywords = []
        emotion_keywords = self.emotion_keywords.get(emotion_type, [])

        for keyword in emotion_keywords:
            if keyword in text:
                keywords.append(keyword)

        return list(set(keywords))[:10]  # 최대 10개

    def _extract_context_clues(self, text: str) -> List[str]:
        """문맥 단서 추출"""
        if not self.config.enable_context_analysis:
            return []

        clues = []

        # 시간 표현
        time_patterns = [
            r'오늘', r'어제', r'내일', r'지금', r'방금', r'나중에',
            r'\d+시', r'\d+분', r'아침', r'점심', r'저녁', r'밤'
        ]
        for pattern in time_patterns:
            if re.search(pattern, text):
                clues.append(f"시간: {pattern}")

        # 장소 표현
        place_patterns = [
            r'집', r'학교', r'회사', r'카페', r'식당', r'병원',
            r'지하철', r'버스', r'차', r'길'
        ]
        for pattern in place_patterns:
            if re.search(pattern, text):
                clues.append(f"장소: {pattern}")

        # 인물 관계
        person_patterns = [
            r'친구', r'가족', r'부모', r'형제', r'동료', r'선생님',
            r'의사', r'사장', r'고객'
        ]
        for pattern in person_patterns:
            if re.search(pattern, text):
                clues.append(f"인물: {pattern}")

        return clues[:5]  # 최대 5개

    def _create_neutral_result(self, text: str, error: Optional[str] = None) -> SentimentResult:
        """중립 결과 생성"""
        return SentimentResult(
            text=text,
            emotion_type=EmotionType.NEUTRAL,
            sentiment_score=SentimentScore(neutral=1.0, confidence=0.5),
            detected_keywords=[],
            emotion_intensity=0.0,
            context_clues=[f"오류: {error}"] if error else []
        )

    def _save_emotion_history(self, user_id: str, result: SentimentResult):
        """감정 히스토리 저장"""
        if user_id not in self.emotion_history:
            self.emotion_history[user_id] = []

        self.emotion_history[user_id].append(result)

        # 최근 100개만 유지
        if len(self.emotion_history[user_id]) > 100:
            self.emotion_history[user_id] = self.emotion_history[user_id][-100:]

    async def analyze_emotion_trend(self, user_id: str, days: int = 7) -> Optional[SentimentTrend]:
        """감정 추세 분석"""
        if user_id not in self.emotion_history:
            return None

        cutoff_time = current_timestamp() - (days * 24 * 3600)
        recent_emotions = [
            result for result in self.emotion_history[user_id]
            if result.timestamp >= cutoff_time
        ]

        if not recent_emotions:
            return None

        # 주요 감정들 추출
        emotion_counts = {}
        total_intensity = 0.0

        for result in recent_emotions:
            emotion = result.emotion_type
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_intensity += result.emotion_intensity

        # 주요 감정 정렬
        dominant_emotions = sorted(emotion_counts.keys(), key=emotion_counts.get, reverse=True)

        # 추세 방향 계산
        if len(recent_emotions) >= 3:
            recent_intensity = sum(r.emotion_intensity for r in recent_emotions[-3:]) / 3
            earlier_intensity = sum(r.emotion_intensity for r in recent_emotions[:-3]) / max(len(recent_emotions) - 3, 1)

            if recent_intensity > earlier_intensity + 0.1:
                trend_direction = 'improving'
            elif recent_intensity < earlier_intensity - 0.1:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'

        return SentimentTrend(
            user_id=user_id,
            time_window=timedelta(days=days),
            emotion_history=recent_emotions,
            dominant_emotions=dominant_emotions[:5],
            trend_direction=trend_direction,
            average_intensity=total_intensity / len(recent_emotions)
        )

    def get_emotion_statistics(self, user_id: str) -> Dict[str, Any]:
        """감정 통계 반환"""
        if user_id not in self.emotion_history:
            return {}

        history = self.emotion_history[user_id]
        if not history:
            return {}

        # 감정별 분포
        emotion_distribution = {}
        for result in history:
            emotion = result.emotion_type.value
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1

        # 평균 감정 강도
        avg_intensity = sum(r.emotion_intensity for r in history) / len(history)

        # 최근 감정
        recent_emotion = history[-1].emotion_type.value if history else 'neutral'

        return {
            'total_analyses': len(history),
            'emotion_distribution': emotion_distribution,
            'average_intensity': avg_intensity,
            'recent_emotion': recent_emotion,
            'dominant_emotion': max(emotion_distribution, key=emotion_distribution.get)
        }

class ContextAnalyzer:
    """문맥 분석기"""

    def __init__(self):
        self.logger = create_logger(__name__)

    def analyze_context(self, text: str, previous_results: List[SentimentResult]) -> Dict[str, Any]:
        """문맥 분석"""
        context = {
            'conversation_flow': self._analyze_conversation_flow(previous_results),
            'topic_consistency': self._analyze_topic_consistency(text, previous_results),
            'emotional_progression': self._analyze_emotional_progression(previous_results)
        }

        return context

    def _analyze_conversation_flow(self, previous_results: List[SentimentResult]) -> str:
        """대화 흐름 분석"""
        if len(previous_results) < 2:
            return 'insufficient_data'

        # 감정 변화 패턴 분석
        recent_emotions = [r.emotion_type for r in previous_results[-5:]]

        if len(set(recent_emotions)) == 1:
            return 'consistent'
        elif len(recent_emotions) >= 3:
            if recent_emotions[-1] != recent_emotions[-2] != recent_emotions[-3]:
                return 'fluctuating'

        return 'transitioning'

    def _analyze_topic_consistency(self, current_text: str, previous_results: List[SentimentResult]) -> float:
        """주제 일관성 분석"""
        if not previous_results:
            return 0.0

        # 간단한 키워드 기반 유사도 계산
        current_keywords = set(current_text.split())

        similarities = []
        for prev_result in previous_results[-3:]:  # 최근 3개만
            prev_keywords = set(prev_result.text.split())
            if prev_keywords:
                similarity = len(current_keywords & prev_keywords) / len(current_keywords | prev_keywords)
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _analyze_emotional_progression(self, previous_results: List[SentimentResult]) -> Dict[str, Any]:
        """감정 진행 분석"""
        if len(previous_results) < 2:
            return {'trend': 'unknown', 'stability': 0.0}

        intensities = [r.emotion_intensity for r in previous_results[-10:]]

        # 추세 계산
        if len(intensities) >= 3:
            recent_avg = sum(intensities[-3:]) / 3
            earlier_avg = sum(intensities[:-3]) / max(len(intensities) - 3, 1)

            if recent_avg > earlier_avg + 0.1:
                trend = 'increasing'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'

        # 안정성 계산 (표준편차의 역수)
        if len(intensities) > 1:
            mean_intensity = sum(intensities) / len(intensities)
            variance = sum((x - mean_intensity) ** 2 for x in intensities) / len(intensities)
            stability = 1.0 / (1.0 + variance)
        else:
            stability = 1.0

        return {
            'trend': trend,
            'stability': stability,
            'average_intensity': sum(intensities) / len(intensities)
        }