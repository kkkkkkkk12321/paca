"""
Module: integrations.nlp.semantic_analyzer
Purpose: Korean semantic analysis with sentiment, emotion, and meaning extraction
Author: PACA Development Team
Created: 2024-09-24
Last Modified: 2024-09-24
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
import re

from .morphology_analyzer import MorphologyAnalyzer
from .syntax_parser import SyntaxParser, SyntacticTree

logger = logging.getLogger(__name__)

class SentimentPolarity(Enum):
    """Sentiment polarity categories."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class EmotionType(Enum):
    """Korean emotion categories."""
    JOY = "joy"           # 기쁨
    SADNESS = "sadness"   # 슬픔
    ANGER = "anger"       # 분노
    FEAR = "fear"         # 두려움
    SURPRISE = "surprise" # 놀라움
    DISGUST = "disgust"   # 혐오
    TRUST = "trust"       # 신뢰
    ANTICIPATION = "anticipation" # 기대

class SemanticRole(Enum):
    """Semantic roles in Korean."""
    AGENT = "agent"           # 행위자
    PATIENT = "patient"       # 대상
    THEME = "theme"           # 주제
    INSTRUMENT = "instrument" # 도구
    LOCATION = "location"     # 장소
    TIME = "time"            # 시간
    MANNER = "manner"        # 방법
    PURPOSE = "purpose"      # 목적

@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    polarity: SentimentPolarity
    intensity: float  # 0.0 to 1.0
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float

@dataclass
class EmotionResult:
    """Emotion analysis result."""
    primary_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    intensity: float
    confidence: float

@dataclass
class SemanticEntity:
    """Semantic entity in text."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    semantic_role: Optional[SemanticRole]
    features: Dict[str, Any]

@dataclass
class SemanticRelation:
    """Semantic relation between entities."""
    relation_type: str
    subject: SemanticEntity
    object: SemanticEntity
    confidence: float

@dataclass
class SemanticAnalysisResult:
    """Complete semantic analysis result."""
    text: str
    sentiment: SentimentResult
    emotion: EmotionResult
    entities: List[SemanticEntity]
    relations: List[SemanticRelation]
    semantic_roles: Dict[str, SemanticRole]
    key_concepts: List[str]
    confidence: float

class SemanticAnalyzer:
    """
    Advanced Korean semantic analyzer with sentiment, emotion, and meaning analysis.

    Provides comprehensive semantic analysis including sentiment analysis,
    emotion recognition, semantic role labeling, and concept extraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic analyzer.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.morphology_analyzer = MorphologyAnalyzer()
        self.syntax_parser = SyntaxParser()

        # Load semantic resources
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.semantic_patterns = self._load_semantic_patterns()
        self.concept_extractors = self._load_concept_extractors()

        self._initialized = False

    def _load_sentiment_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load Korean sentiment lexicon."""
        return {
            # Positive words
            '좋다': {'positive': 0.8, 'negative': 0.0, 'neutral': 0.2},
            '훌륭하다': {'positive': 0.9, 'negative': 0.0, 'neutral': 0.1},
            '멋지다': {'positive': 0.85, 'negative': 0.0, 'neutral': 0.15},
            '아름답다': {'positive': 0.8, 'negative': 0.0, 'neutral': 0.2},
            '행복하다': {'positive': 0.9, 'negative': 0.0, 'neutral': 0.1},
            '즐겁다': {'positive': 0.85, 'negative': 0.0, 'neutral': 0.15},
            '기쁘다': {'positive': 0.9, 'negative': 0.0, 'neutral': 0.1},
            '사랑하다': {'positive': 0.95, 'negative': 0.0, 'neutral': 0.05},
            '고맙다': {'positive': 0.8, 'negative': 0.0, 'neutral': 0.2},
            '감사하다': {'positive': 0.85, 'negative': 0.0, 'neutral': 0.15},

            # Negative words
            '나쁘다': {'positive': 0.0, 'negative': 0.8, 'neutral': 0.2},
            '싫다': {'positive': 0.0, 'negative': 0.85, 'neutral': 0.15},
            '화나다': {'positive': 0.0, 'negative': 0.9, 'neutral': 0.1},
            '슬프다': {'positive': 0.0, 'negative': 0.9, 'neutral': 0.1},
            '무섭다': {'positive': 0.0, 'negative': 0.85, 'neutral': 0.15},
            '끔찍하다': {'positive': 0.0, 'negative': 0.95, 'neutral': 0.05},
            '미안하다': {'positive': 0.0, 'negative': 0.7, 'neutral': 0.3},
            '죄송하다': {'positive': 0.0, 'negative': 0.6, 'neutral': 0.4},
            '실망하다': {'positive': 0.0, 'negative': 0.8, 'neutral': 0.2},

            # Neutral words
            '그렇다': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            '있다': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            '하다': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            '되다': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            '같다': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},

            # Intensifiers
            '매우': {'intensifier': 1.5},
            '아주': {'intensifier': 1.4},
            '정말': {'intensifier': 1.3},
            '너무': {'intensifier': 1.6},
            '완전': {'intensifier': 1.7},
            '진짜': {'intensifier': 1.2},
            '정말로': {'intensifier': 1.4},

            # Diminishers
            '조금': {'intensifier': 0.7},
            '약간': {'intensifier': 0.6},
            '살짝': {'intensifier': 0.5},
            '좀': {'intensifier': 0.8},
        }

    def _load_emotion_lexicon(self) -> Dict[str, Dict[EmotionType, float]]:
        """Load Korean emotion lexicon."""
        return {
            # Joy emotions
            '기쁘다': {EmotionType.JOY: 0.9, EmotionType.SURPRISE: 0.1},
            '즐겁다': {EmotionType.JOY: 0.8, EmotionType.ANTICIPATION: 0.2},
            '행복하다': {EmotionType.JOY: 0.95, EmotionType.TRUST: 0.05},
            '웃다': {EmotionType.JOY: 0.7, EmotionType.SURPRISE: 0.3},

            # Sadness emotions
            '슬프다': {EmotionType.SADNESS: 0.9, EmotionType.FEAR: 0.1},
            '우울하다': {EmotionType.SADNESS: 0.8, EmotionType.DISGUST: 0.2},
            '울다': {EmotionType.SADNESS: 0.85, EmotionType.ANGER: 0.15},
            '외롭다': {EmotionType.SADNESS: 0.75, EmotionType.FEAR: 0.25},

            # Anger emotions
            '화나다': {EmotionType.ANGER: 0.9, EmotionType.DISGUST: 0.1},
            '짜증나다': {EmotionType.ANGER: 0.8, EmotionType.DISGUST: 0.2},
            '분하다': {EmotionType.ANGER: 0.85, EmotionType.SADNESS: 0.15},
            '격분하다': {EmotionType.ANGER: 0.95, EmotionType.FEAR: 0.05},

            # Fear emotions
            '무섭다': {EmotionType.FEAR: 0.9, EmotionType.SURPRISE: 0.1},
            '두렵다': {EmotionType.FEAR: 0.85, EmotionType.SADNESS: 0.15},
            '걱정되다': {EmotionType.FEAR: 0.7, EmotionType.SADNESS: 0.3},
            '불안하다': {EmotionType.FEAR: 0.8, EmotionType.ANGER: 0.2},

            # Surprise emotions
            '놀라다': {EmotionType.SURPRISE: 0.9, EmotionType.FEAR: 0.1},
            '신기하다': {EmotionType.SURPRISE: 0.7, EmotionType.JOY: 0.3},
            '의외다': {EmotionType.SURPRISE: 0.8, EmotionType.TRUST: 0.2},

            # Disgust emotions
            '싫다': {EmotionType.DISGUST: 0.8, EmotionType.ANGER: 0.2},
            '역겹다': {EmotionType.DISGUST: 0.9, EmotionType.FEAR: 0.1},
            '끔찍하다': {EmotionType.DISGUST: 0.85, EmotionType.FEAR: 0.15},

            # Trust emotions
            '믿다': {EmotionType.TRUST: 0.9, EmotionType.JOY: 0.1},
            '신뢰하다': {EmotionType.TRUST: 0.95, EmotionType.ANTICIPATION: 0.05},
            '의지하다': {EmotionType.TRUST: 0.8, EmotionType.ANTICIPATION: 0.2},

            # Anticipation emotions
            '기대하다': {EmotionType.ANTICIPATION: 0.9, EmotionType.JOY: 0.1},
            '바라다': {EmotionType.ANTICIPATION: 0.8, EmotionType.TRUST: 0.2},
            '희망하다': {EmotionType.ANTICIPATION: 0.85, EmotionType.JOY: 0.15},
        }

    def _load_semantic_patterns(self) -> Dict[str, List[str]]:
        """Load semantic role patterns."""
        return {
            'agent_markers': ['이', '가', '께서'],  # Subject particles
            'patient_markers': ['을', '를'],        # Object particles
            'location_markers': ['에서', '에', '로', '으로'],  # Location particles
            'time_markers': ['에', '때', '동안'],    # Time markers
            'instrument_markers': ['로', '으로', '으로써'], # Instrument markers
            'manner_markers': ['게', '히', '으로'],  # Manner markers
        }

    def _load_concept_extractors(self) -> Dict[str, re.Pattern]:
        """Load concept extraction patterns."""
        return {
            'person': re.compile(r'[가-힣]+(?:씨|님|군|양|선생|교수|박사|회장|사장|부장)'),
            'organization': re.compile(r'[가-힣]+(?:회사|기업|단체|협회|학교|대학|연구소)'),
            'location': re.compile(r'[가-힣]+(?:시|도|구|군|동|읍|면|리|역|공항|병원)'),
            'time': re.compile(r'(?:\d+년|\d+월|\d+일|\d+시|\d+분|오늘|내일|어제|지금)'),
            'number': re.compile(r'\d+(?:개|명|마리|대|권|장|병|잔|그릇)'),
        }

    async def initialize(self) -> bool:
        """
        Initialize the semantic analyzer.

        Returns:
            bool: True if initialization successful
        """
        try:
            await self.morphology_analyzer.initialize()
            await self.syntax_parser.initialize()
            self._initialized = True
            logger.info("Semantic analyzer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            return False

    async def analyze(self,
                     text: str,
                     include_sentiment: bool = True,
                     include_emotion: bool = True,
                     include_entities: bool = True,
                     include_relations: bool = True) -> SemanticAnalysisResult:
        """
        Perform comprehensive semantic analysis.

        Args:
            text: Korean text to analyze
            include_sentiment: Include sentiment analysis
            include_emotion: Include emotion analysis
            include_entities: Include entity extraction
            include_relations: Include relation extraction

        Returns:
            SemanticAnalysisResult: Complete semantic analysis
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Get syntactic analysis first
            syntactic_tree = await self.syntax_parser.parse(text)

            # Analyze sentiment
            sentiment = None
            if include_sentiment:
                sentiment = await self._analyze_sentiment(text, syntactic_tree)

            # Analyze emotion
            emotion = None
            if include_emotion:
                emotion = await self._analyze_emotion(text, syntactic_tree)

            # Extract entities
            entities = []
            if include_entities:
                entities = await self._extract_entities(text, syntactic_tree)

            # Extract relations
            relations = []
            if include_relations and entities:
                relations = await self._extract_relations(entities, syntactic_tree)

            # Assign semantic roles
            semantic_roles = await self._assign_semantic_roles(syntactic_tree)

            # Extract key concepts
            key_concepts = await self._extract_key_concepts(text, entities)

            # Calculate overall confidence
            confidence = self._calculate_semantic_confidence(
                sentiment, emotion, entities, relations
            )

            result = SemanticAnalysisResult(
                text=text,
                sentiment=sentiment,
                emotion=emotion,
                entities=entities,
                relations=relations,
                semantic_roles=semantic_roles,
                key_concepts=key_concepts,
                confidence=confidence
            )

            processing_time = time.time() - start_time
            logger.debug(f"Semantic analysis completed in {processing_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            raise

    async def _analyze_sentiment(self, text: str, syntactic_tree: SyntacticTree) -> SentimentResult:
        """
        Analyze sentiment of the text.

        Args:
            text: Text to analyze
            syntactic_tree: Syntactic parse

        Returns:
            SentimentResult: Sentiment analysis result
        """
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        total_words = 0

        current_intensifier = 1.0

        for node in syntactic_tree.nodes:
            lemma = node.morphology.lemma.lower()

            # Check for intensifiers first
            if lemma in self.sentiment_lexicon and 'intensifier' in self.sentiment_lexicon[lemma]:
                current_intensifier = self.sentiment_lexicon[lemma]['intensifier']
                continue

            # Check for sentiment words
            if lemma in self.sentiment_lexicon:
                scores = self.sentiment_lexicon[lemma]

                if 'positive' in scores:
                    positive_score += scores['positive'] * current_intensifier
                    negative_score += scores.get('negative', 0.0) * current_intensifier
                    neutral_score += scores.get('neutral', 0.0)
                    total_words += 1

                # Reset intensifier after applying
                current_intensifier = 1.0

        # Normalize scores
        if total_words > 0:
            positive_score /= total_words
            negative_score /= total_words
            neutral_score /= total_words

            # Ensure scores sum to 1
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
        else:
            # Default to neutral if no sentiment words found
            neutral_score = 1.0

        # Determine polarity
        max_score = max(positive_score, negative_score, neutral_score)
        if max_score == positive_score:
            if positive_score > 0.8:
                polarity = SentimentPolarity.VERY_POSITIVE
            else:
                polarity = SentimentPolarity.POSITIVE
        elif max_score == negative_score:
            if negative_score > 0.8:
                polarity = SentimentPolarity.VERY_NEGATIVE
            else:
                polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL

        # Calculate intensity and confidence
        intensity = max_score
        confidence = max_score if total_words > 0 else 0.5  # Lower confidence for no sentiment words

        return SentimentResult(
            polarity=polarity,
            intensity=intensity,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score
        )

    async def _analyze_emotion(self, text: str, syntactic_tree: SyntacticTree) -> EmotionResult:
        """
        Analyze emotions in the text.

        Args:
            text: Text to analyze
            syntactic_tree: Syntactic parse

        Returns:
            EmotionResult: Emotion analysis result
        """
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}
        total_words = 0

        for node in syntactic_tree.nodes:
            lemma = node.morphology.lemma.lower()

            if lemma in self.emotion_lexicon:
                emotions = self.emotion_lexicon[lemma]

                for emotion_type, score in emotions.items():
                    emotion_scores[emotion_type] += score

                total_words += 1

        # Normalize scores
        if total_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_words

        # Find primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion_type = primary_emotion[0]
        intensity = primary_emotion[1]

        # Calculate confidence
        confidence = intensity if total_words > 0 else 0.3

        return EmotionResult(
            primary_emotion=primary_emotion_type,
            emotion_scores=emotion_scores,
            intensity=intensity,
            confidence=confidence
        )

    async def _extract_entities(self, text: str, syntactic_tree: SyntacticTree) -> List[SemanticEntity]:
        """
        Extract semantic entities from text.

        Args:
            text: Original text
            syntactic_tree: Syntactic parse

        Returns:
            List[SemanticEntity]: Extracted entities
        """
        entities = []

        # Extract using pattern matching
        for entity_type, pattern in self.concept_extractors.items():
            for match in pattern.finditer(text):
                entity = SemanticEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    semantic_role=None,  # Will be assigned later
                    features={'pattern_based': True}
                )
                entities.append(entity)

        # Extract from syntactic analysis
        for node in syntactic_tree.nodes:
            if node.morphology.word_class.value == 'noun':
                # Check if it's already covered by pattern matching
                existing = any(
                    entity.start_pos <= node.id <= entity.end_pos
                    for entity in entities
                )

                if not existing:
                    entity = SemanticEntity(
                        text=node.text,
                        entity_type='noun',
                        start_pos=node.id,
                        end_pos=node.id + len(node.text),
                        semantic_role=self._map_syntactic_to_semantic_role(node.syntactic_role),
                        features={
                            'pos': node.morphology.pos_tag,
                            'syntactic_role': node.syntactic_role.value if node.syntactic_role else None
                        }
                    )
                    entities.append(entity)

        return entities

    def _map_syntactic_to_semantic_role(self, syntactic_role) -> Optional[SemanticRole]:
        """Map syntactic role to semantic role."""
        if not syntactic_role:
            return None

        mapping = {
            'subject': SemanticRole.AGENT,
            'object': SemanticRole.PATIENT,
            'modifier': SemanticRole.THEME,
            'adverbial': SemanticRole.MANNER,
        }

        return mapping.get(syntactic_role.value)

    async def _extract_relations(self,
                               entities: List[SemanticEntity],
                               syntactic_tree: SyntacticTree) -> List[SemanticRelation]:
        """
        Extract semantic relations between entities.

        Args:
            entities: Extracted entities
            syntactic_tree: Syntactic parse

        Returns:
            List[SemanticRelation]: Semantic relations
        """
        relations = []

        # Extract relations based on syntactic dependencies
        for dependency in syntactic_tree.dependencies:
            head_entity = self._find_entity_by_position(entities, dependency['head'])
            dependent_entity = self._find_entity_by_position(entities, dependency['dependent'])

            if head_entity and dependent_entity:
                relation = SemanticRelation(
                    relation_type=dependency['relation'],
                    subject=dependent_entity,
                    object=head_entity,
                    confidence=0.8  # Base confidence for syntax-based relations
                )
                relations.append(relation)

        # Extract relations based on common patterns
        # This is a simplified implementation - real relation extraction is more complex
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Simple proximity-based relation
                if abs(entity1.start_pos - entity2.start_pos) < 5:  # Nearby entities
                    relation_type = self._infer_relation_type(entity1, entity2)
                    if relation_type:
                        relation = SemanticRelation(
                            relation_type=relation_type,
                            subject=entity1,
                            object=entity2,
                            confidence=0.6  # Lower confidence for inferred relations
                        )
                        relations.append(relation)

        return relations

    def _find_entity_by_position(self, entities: List[SemanticEntity], position: int) -> Optional[SemanticEntity]:
        """Find entity at given position."""
        for entity in entities:
            if entity.start_pos <= position <= entity.end_pos:
                return entity
        return None

    def _infer_relation_type(self, entity1: SemanticEntity, entity2: SemanticEntity) -> Optional[str]:
        """Infer relation type between two entities."""
        # Simple heuristics based on entity types
        if entity1.entity_type == 'person' and entity2.entity_type == 'organization':
            return 'member_of'
        elif entity1.entity_type == 'person' and entity2.entity_type == 'location':
            return 'located_at'
        elif entity1.entity_type == 'organization' and entity2.entity_type == 'location':
            return 'based_in'

        return None

    async def _assign_semantic_roles(self, syntactic_tree: SyntacticTree) -> Dict[str, SemanticRole]:
        """
        Assign semantic roles to syntactic constituents.

        Args:
            syntactic_tree: Syntactic parse

        Returns:
            Dict[str, SemanticRole]: Semantic role assignments
        """
        semantic_roles = {}

        for node in syntactic_tree.nodes:
            if node.syntactic_role:
                semantic_role = self._map_syntactic_to_semantic_role(node.syntactic_role)
                if semantic_role:
                    semantic_roles[node.text] = semantic_role

        return semantic_roles

    async def _extract_key_concepts(self, text: str, entities: List[SemanticEntity]) -> List[str]:
        """
        Extract key concepts from text.

        Args:
            text: Original text
            entities: Extracted entities

        Returns:
            List[str]: Key concepts
        """
        key_concepts = []

        # Add important entities as key concepts
        important_types = {'person', 'organization', 'location'}
        for entity in entities:
            if entity.entity_type in important_types:
                key_concepts.append(entity.text)

        # Add frequent nouns as potential concepts
        # This is simplified - real concept extraction would use TF-IDF, topic modeling, etc.
        words = text.split()
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add frequent words as concepts
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for word, freq in frequent_words:
            if freq > 1 and word not in key_concepts:
                key_concepts.append(word)

        return key_concepts[:10]  # Limit to top 10 concepts

    def _calculate_semantic_confidence(self,
                                     sentiment: Optional[SentimentResult],
                                     emotion: Optional[EmotionResult],
                                     entities: List[SemanticEntity],
                                     relations: List[SemanticRelation]) -> float:
        """
        Calculate overall confidence for semantic analysis.

        Args:
            sentiment: Sentiment analysis result
            emotion: Emotion analysis result
            entities: Extracted entities
            relations: Extracted relations

        Returns:
            float: Overall confidence score
        """
        confidences = []

        if sentiment:
            confidences.append(sentiment.confidence)

        if emotion:
            confidences.append(emotion.confidence)

        if entities:
            # Confidence based on entity extraction success
            pattern_entities = [e for e in entities if e.features.get('pattern_based')]
            entity_confidence = len(pattern_entities) / max(len(entities), 1)
            confidences.append(entity_confidence)

        if relations:
            # Confidence based on relation extraction
            avg_relation_confidence = sum(r.confidence for r in relations) / len(relations)
            confidences.append(avg_relation_confidence)

        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default confidence

    async def get_semantic_statistics(self, result: SemanticAnalysisResult) -> Dict[str, Any]:
        """
        Get statistics for semantic analysis result.

        Args:
            result: Semantic analysis result

        Returns:
            Dict[str, Any]: Statistics
        """
        stats = {
            'text_length': len(result.text),
            'overall_confidence': result.confidence,
            'sentiment': {
                'polarity': result.sentiment.polarity.value if result.sentiment else None,
                'intensity': result.sentiment.intensity if result.sentiment else 0,
                'confidence': result.sentiment.confidence if result.sentiment else 0
            },
            'emotion': {
                'primary': result.emotion.primary_emotion.value if result.emotion else None,
                'intensity': result.emotion.intensity if result.emotion else 0,
                'confidence': result.emotion.confidence if result.emotion else 0
            },
            'entities': {
                'total_count': len(result.entities),
                'by_type': {}
            },
            'relations': {
                'total_count': len(result.relations),
                'by_type': {}
            },
            'key_concepts_count': len(result.key_concepts)
        }

        # Count entities by type
        for entity in result.entities:
            entity_type = entity.entity_type
            stats['entities']['by_type'][entity_type] = \
                stats['entities']['by_type'].get(entity_type, 0) + 1

        # Count relations by type
        for relation in result.relations:
            relation_type = relation.relation_type
            stats['relations']['by_type'][relation_type] = \
                stats['relations']['by_type'].get(relation_type, 0) + 1

        return stats