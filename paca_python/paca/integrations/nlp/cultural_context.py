"""
Module: integrations.nlp.cultural_context
Purpose: Korean cultural context analysis with honorifics, social relationships, and cultural nuances
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

logger = logging.getLogger(__name__)

class HonorificLevel(Enum):
    """Korean honorific levels with cultural context."""
    CASUAL = "casual"               # 반말 (casual speech)
    POLITE_INFORMAL = "polite_informal"  # 해요체 (polite informal)
    POLITE_FORMAL = "polite_formal"      # 합니다체 (polite formal)
    HUMBLE = "humble"               # 겸양어 (humble language)
    ELEVATED = "elevated"           # 높임어 (elevated language)
    SUPER_ELEVATED = "super_elevated"    # 극존칭 (super elevated)

class SocialRelationship(Enum):
    """Social relationship types in Korean culture."""
    FAMILY_ELDER = "family_elder"   # 가족 윗사람
    FAMILY_YOUNGER = "family_younger" # 가족 아랫사람
    SOCIAL_SUPERIOR = "social_superior" # 사회적 윗사람
    SOCIAL_SUBORDINATE = "social_subordinate" # 사회적 아랫사람
    PEER = "peer"                   # 동급자
    STRANGER_OLDER = "stranger_older" # 모르는 윗사람
    STRANGER_YOUNGER = "stranger_younger" # 모르는 아랫사람
    PROFESSIONAL = "professional"   # 업무 관계

class CulturalConcept(Enum):
    """Korean cultural concepts."""
    NUNCHI = "nunchi"               # 눈치 (social awareness)
    JEONG = "jeong"                 # 정 (affection/attachment)
    HAN = "han"                     # 한 (collective sorrow)
    AEGYO = "aegyo"                 # 애교 (cute charm)
    CHEMYON = "chemyon"             # 체면 (face/reputation)
    URI = "uri"                     # 우리 (we/us - collective identity)
    JIREUGI = "jireugi"            # 지르기 (persistence)

@dataclass
class CulturalMarker:
    """Cultural marker in text."""
    text: str
    marker_type: str
    cultural_significance: str
    start_pos: int
    end_pos: int
    confidence: float

@dataclass
class HonorificAnalysis:
    """Honorific system analysis."""
    overall_level: HonorificLevel
    consistency_score: float
    specific_markers: List[CulturalMarker]
    relationship_indicators: List[str]
    appropriateness_score: float

@dataclass
class SocialContextAnalysis:
    """Social context analysis."""
    inferred_relationship: SocialRelationship
    power_distance: float  # -1 (subordinate) to 1 (superior)
    formality_level: float  # 0 (informal) to 1 (formal)
    social_markers: List[CulturalMarker]
    confidence: float

@dataclass
class CulturalContextResult:
    """Complete cultural context analysis."""
    text: str
    honorific_analysis: HonorificAnalysis
    social_context: SocialContextAnalysis
    cultural_concepts: List[str]
    cultural_markers: List[CulturalMarker]
    appropriateness_suggestions: List[str]
    overall_confidence: float

class CulturalContextProcessor:
    """
    Korean cultural context processor for analyzing social relationships,
    honorific usage, and cultural nuances in Korean text.

    Provides comprehensive cultural analysis including honorific appropriateness,
    social relationship inference, and cultural concept detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cultural context processor.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        # Load cultural linguistic resources
        self.honorific_markers = self._load_honorific_markers()
        self.relationship_indicators = self._load_relationship_indicators()
        self.cultural_expressions = self._load_cultural_expressions()
        self.formality_patterns = self._load_formality_patterns()

        # Load cultural knowledge base
        self.cultural_concepts_kb = self._load_cultural_concepts()
        self.social_situations = self._load_social_situations()

        self._initialized = False

    def _load_honorific_markers(self) -> Dict[str, Dict[str, Any]]:
        """Load Korean honorific markers and their cultural significance."""
        return {
            # Verb forms
            '하십니다': {
                'level': HonorificLevel.POLITE_FORMAL,
                'type': 'formal_verb',
                'cultural_weight': 0.9,
                'appropriateness': ['business', 'formal_presentation', 'elderly']
            },
            '하세요': {
                'level': HonorificLevel.POLITE_INFORMAL,
                'type': 'polite_verb',
                'cultural_weight': 0.7,
                'appropriateness': ['general_polite', 'service', 'acquaintance']
            },
            '해요': {
                'level': HonorificLevel.POLITE_INFORMAL,
                'type': 'polite_verb',
                'cultural_weight': 0.6,
                'appropriateness': ['casual_polite', 'friend_polite']
            },
            '해': {
                'level': HonorificLevel.CASUAL,
                'type': 'casual_verb',
                'cultural_weight': 0.3,
                'appropriateness': ['close_friend', 'younger_person', 'family']
            },

            # Honorific particles and titles
            '께서': {
                'level': HonorificLevel.ELEVATED,
                'type': 'subject_honorific',
                'cultural_weight': 0.8,
                'replaces': '이/가'
            },
            '께': {
                'level': HonorificLevel.ELEVATED,
                'type': 'dative_honorific',
                'cultural_weight': 0.8,
                'replaces': '에게'
            },
            '님': {
                'level': HonorificLevel.ELEVATED,
                'type': 'title_honorific',
                'cultural_weight': 0.7,
                'usage': 'after_name'
            },
            '씨': {
                'level': HonorificLevel.POLITE_INFORMAL,
                'type': 'title_honorific',
                'cultural_weight': 0.5,
                'usage': 'after_name'
            },

            # Humble forms
            '드리다': {
                'level': HonorificLevel.HUMBLE,
                'type': 'humble_verb',
                'cultural_weight': 0.8,
                'replaces': '주다'
            },
            '올리다': {
                'level': HonorificLevel.HUMBLE,
                'type': 'humble_verb',
                'cultural_weight': 0.7,
                'replaces': '주다/보내다'
            },
            '여쭙다': {
                'level': HonorificLevel.HUMBLE,
                'type': 'humble_verb',
                'cultural_weight': 0.9,
                'replaces': '묻다'
            },

            # Super elevated forms
            '하옵니다': {
                'level': HonorificLevel.SUPER_ELEVATED,
                'type': 'archaic_formal',
                'cultural_weight': 1.0,
                'context': 'ceremonial'
            },
            '계시다': {
                'level': HonorificLevel.ELEVATED,
                'type': 'existential_honorific',
                'cultural_weight': 0.8,
                'replaces': '있다'
            }
        }

    def _load_relationship_indicators(self) -> Dict[str, List[str]]:
        """Load indicators of social relationships."""
        return {
            'family_terms': [
                '아버지', '어머니', '아버님', '어머님', '할아버지', '할머니',
                '삼촌', '이모', '고모', '외삼촌', '형', '누나', '언니', '오빠',
                '동생', '아들', '딸', '손자', '손녀'
            ],
            'professional_titles': [
                '선생님', '교수님', '박사님', '사장님', '회장님', '부장님',
                '과장님', '팀장님', '대표님', '원장님', '소장님'
            ],
            'formal_titles': [
                '각하', '폐하', '전하', '귀하', '님', '씨', '양', '군'
            ],
            'casual_addresses': [
                '야', '너', '얘', '쟤', '친구', '자기'
            ],
            'age_indicators': [
                '연상', '연하', '선배', '후배', '동갑', '나이', '살'
            ]
        }

    def _load_cultural_expressions(self) -> Dict[str, Dict[str, Any]]:
        """Load Korean cultural expressions and idioms."""
        return {
            # Nunchi (social awareness)
            '눈치': {
                'concept': CulturalConcept.NUNCHI,
                'variants': ['눈치보다', '눈치없다', '눈치채다'],
                'cultural_note': 'Ability to read social situations and respond appropriately'
            },
            '눈치보다': {
                'concept': CulturalConcept.NUNCHI,
                'meaning': 'being cautious about others\' reactions',
                'usage': 'social_awareness'
            },

            # Jeong (affection/attachment)
            '정': {
                'concept': CulturalConcept.JEONG,
                'variants': ['정이 들다', '정이 많다', '정을 주다'],
                'cultural_note': 'Deep emotional bond and affection'
            },
            '정이 들다': {
                'concept': CulturalConcept.JEONG,
                'meaning': 'developing emotional attachment',
                'usage': 'relationship_building'
            },

            # Han (collective sorrow)
            '한': {
                'concept': CulturalConcept.HAN,
                'variants': ['한을 품다', '한이 서리다'],
                'cultural_note': 'Deep-rooted sorrow or resentment'
            },

            # Chemyon (face/reputation)
            '체면': {
                'concept': CulturalConcept.CHEMYON,
                'variants': ['체면을 지키다', '체면을 세우다', '체면이 서다'],
                'cultural_note': 'Social face and reputation'
            },
            '면목': {
                'concept': CulturalConcept.CHEMYON,
                'meaning': 'dignity, honor',
                'usage': 'social_status'
            },

            # Uri (collective identity)
            '우리': {
                'concept': CulturalConcept.URI,
                'meaning': 'collective we/us',
                'cultural_note': 'Strong in-group identity',
                'usage': 'group_belonging'
            },

            # Common cultural phrases
            '고생하셨습니다': {
                'concept': 'respect_for_effort',
                'cultural_note': 'Acknowledging someone\'s hard work',
                'appropriateness': 'workplace_respect'
            },
            '수고하세요': {
                'concept': 'workplace_courtesy',
                'cultural_note': 'Casual workplace farewell',
                'appropriateness': 'colleague_interaction'
            }
        }

    def _load_formality_patterns(self) -> Dict[str, float]:
        """Load formality patterns with scores."""
        return {
            # Formal patterns (higher scores)
            r'습니다$': 0.9,
            r'습니까$': 0.9,
            r'하십시오$': 1.0,
            r'해주십시오$': 0.95,
            r'께서': 0.8,
            r'님$': 0.7,

            # Polite patterns
            r'해요$': 0.6,
            r'어요$': 0.6,
            r'아요$': 0.6,
            r'세요$': 0.7,
            r'씨$': 0.5,

            # Casual patterns (lower scores)
            r'해$': 0.2,
            r'어$': 0.2,
            r'아$': 0.2,
            r'야$': 0.1,
            r'너$': 0.1,
        }

    def _load_cultural_concepts(self) -> Dict[CulturalConcept, Dict[str, Any]]:
        """Load detailed cultural concepts knowledge base."""
        return {
            CulturalConcept.NUNCHI: {
                'description': 'Social awareness and sensitivity to others\' emotions',
                'importance': 0.9,
                'contexts': ['workplace', 'social_gathering', 'family'],
                'behavioral_implications': ['indirect_communication', 'context_sensitivity']
            },
            CulturalConcept.JEONG: {
                'description': 'Deep emotional bonds and human warmth',
                'importance': 0.8,
                'contexts': ['friendship', 'romance', 'community'],
                'behavioral_implications': ['loyalty', 'emotional_investment']
            },
            CulturalConcept.HAN: {
                'description': 'Collective historical sorrow and resilience',
                'importance': 0.7,
                'contexts': ['literature', 'arts', 'historical_discussion'],
                'behavioral_implications': ['melancholy', 'perseverance']
            },
            CulturalConcept.CHEMYON: {
                'description': 'Social face and reputation maintenance',
                'importance': 0.9,
                'contexts': ['public', 'professional', 'formal'],
                'behavioral_implications': ['dignity_preservation', 'indirect_criticism']
            },
            CulturalConcept.URI: {
                'description': 'Collective identity and in-group belonging',
                'importance': 0.8,
                'contexts': ['family', 'organization', 'nation'],
                'behavioral_implications': ['group_loyalty', 'collective_responsibility']
            }
        }

    def _load_social_situations(self) -> Dict[str, Dict[str, Any]]:
        """Load social situation contexts and appropriate language."""
        return {
            'business_meeting': {
                'required_honorific': HonorificLevel.POLITE_FORMAL,
                'appropriate_markers': ['습니다', '습니까', '님'],
                'avoid_markers': ['해', '야', '너'],
                'cultural_considerations': ['hierarchy_respect', 'formal_distance']
            },
            'casual_conversation': {
                'required_honorific': HonorificLevel.POLITE_INFORMAL,
                'appropriate_markers': ['해요', '어요', '아요'],
                'context_dependent': True
            },
            'family_elder': {
                'required_honorific': HonorificLevel.ELEVATED,
                'appropriate_markers': ['하십니다', '께서', '님'],
                'cultural_considerations': ['filial_respect', 'age_hierarchy']
            },
            'service_interaction': {
                'required_honorific': HonorificLevel.POLITE_FORMAL,
                'appropriate_markers': ['습니다', '세요', '님'],
                'cultural_considerations': ['customer_respect', 'professional_service']
            }
        }

    async def initialize(self) -> bool:
        """
        Initialize the cultural context processor.

        Returns:
            bool: True if initialization successful
        """
        try:
            self._initialized = True
            logger.info("Cultural context processor initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize cultural context processor: {e}")
            return False

    async def analyze_context(self,
                            text: str,
                            context_hint: Optional[str] = None,
                            speaker_info: Optional[Dict[str, Any]] = None,
                            listener_info: Optional[Dict[str, Any]] = None) -> CulturalContextResult:
        """
        Analyze cultural context of Korean text.

        Args:
            text: Korean text to analyze
            context_hint: Optional context hint (e.g., 'business', 'family')
            speaker_info: Speaker information (age, status, etc.)
            listener_info: Listener information (age, status, etc.)

        Returns:
            CulturalContextResult: Complete cultural analysis
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Analyze honorific usage
            honorific_analysis = await self._analyze_honorifics(text)

            # Infer social context
            social_context = await self._analyze_social_context(
                text, context_hint, speaker_info, listener_info
            )

            # Detect cultural concepts
            cultural_concepts = await self._detect_cultural_concepts(text)

            # Find cultural markers
            cultural_markers = await self._extract_cultural_markers(text)

            # Generate appropriateness suggestions
            suggestions = await self._generate_appropriateness_suggestions(
                text, honorific_analysis, social_context
            )

            # Calculate overall confidence
            confidence = self._calculate_cultural_confidence(
                honorific_analysis, social_context, cultural_concepts, cultural_markers
            )

            result = CulturalContextResult(
                text=text,
                honorific_analysis=honorific_analysis,
                social_context=social_context,
                cultural_concepts=cultural_concepts,
                cultural_markers=cultural_markers,
                appropriateness_suggestions=suggestions,
                overall_confidence=confidence
            )

            processing_time = time.time() - start_time
            logger.debug(f"Cultural context analysis completed in {processing_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"Cultural context analysis failed: {e}")
            raise

    async def _analyze_honorifics(self, text: str) -> HonorificAnalysis:
        """
        Analyze honorific usage in the text.

        Args:
            text: Text to analyze

        Returns:
            HonorificAnalysis: Honorific analysis result
        """
        honorific_markers = []
        honorific_levels = []
        total_weight = 0

        # Find honorific markers
        for marker, info in self.honorific_markers.items():
            if marker in text:
                cultural_marker = CulturalMarker(
                    text=marker,
                    marker_type='honorific',
                    cultural_significance=f"{info['type']} - {info['level'].value}",
                    start_pos=text.find(marker),
                    end_pos=text.find(marker) + len(marker),
                    confidence=info['cultural_weight']
                )
                honorific_markers.append(cultural_marker)
                honorific_levels.append(info['level'])
                total_weight += info['cultural_weight']

        # Determine overall level
        if honorific_levels:
            # Weight-based average of honorific levels
            level_weights = {
                HonorificLevel.CASUAL: 0.2,
                HonorificLevel.POLITE_INFORMAL: 0.4,
                HonorificLevel.POLITE_FORMAL: 0.6,
                HonorificLevel.HUMBLE: 0.7,
                HonorificLevel.ELEVATED: 0.8,
                HonorificLevel.SUPER_ELEVATED: 1.0
            }

            weighted_score = sum(
                level_weights[level] * self.honorific_markers[marker]['cultural_weight']
                for level, marker in zip(honorific_levels,
                [m.text for m in honorific_markers])
            ) / total_weight

            # Map score to level
            if weighted_score >= 0.9:
                overall_level = HonorificLevel.SUPER_ELEVATED
            elif weighted_score >= 0.75:
                overall_level = HonorificLevel.ELEVATED
            elif weighted_score >= 0.65:
                overall_level = HonorificLevel.HUMBLE
            elif weighted_score >= 0.5:
                overall_level = HonorificLevel.POLITE_FORMAL
            elif weighted_score >= 0.3:
                overall_level = HonorificLevel.POLITE_INFORMAL
            else:
                overall_level = HonorificLevel.CASUAL
        else:
            overall_level = HonorificLevel.CASUAL
            weighted_score = 0.2

        # Calculate consistency
        consistency_score = self._calculate_honorific_consistency(honorific_levels)

        # Calculate appropriateness (simplified)
        appropriateness_score = min(weighted_score * consistency_score, 1.0)

        # Extract relationship indicators
        relationship_indicators = self._extract_relationship_indicators(text)

        return HonorificAnalysis(
            overall_level=overall_level,
            consistency_score=consistency_score,
            specific_markers=honorific_markers,
            relationship_indicators=relationship_indicators,
            appropriateness_score=appropriateness_score
        )

    def _calculate_honorific_consistency(self, levels: List[HonorificLevel]) -> float:
        """Calculate consistency of honorific usage."""
        if not levels:
            return 1.0

        # Count different levels
        unique_levels = set(levels)

        # Perfect consistency if all same level
        if len(unique_levels) == 1:
            return 1.0

        # Lower consistency for mixed levels
        # Allow some mixing between adjacent levels
        level_order = [
            HonorificLevel.CASUAL,
            HonorificLevel.POLITE_INFORMAL,
            HonorificLevel.POLITE_FORMAL,
            HonorificLevel.HUMBLE,
            HonorificLevel.ELEVATED,
            HonorificLevel.SUPER_ELEVATED
        ]

        level_indices = [level_order.index(level) for level in levels]
        variation = max(level_indices) - min(level_indices)

        # Consistency decreases with variation
        return max(0.0, 1.0 - (variation * 0.15))

    def _extract_relationship_indicators(self, text: str) -> List[str]:
        """Extract relationship indicators from text."""
        indicators = []

        for category, terms in self.relationship_indicators.items():
            for term in terms:
                if term in text:
                    indicators.append(f"{category}: {term}")

        return indicators

    async def _analyze_social_context(self,
                                    text: str,
                                    context_hint: Optional[str],
                                    speaker_info: Optional[Dict[str, Any]],
                                    listener_info: Optional[Dict[str, Any]]) -> SocialContextAnalysis:
        """
        Analyze social context and relationships.

        Args:
            text: Text to analyze
            context_hint: Context hint
            speaker_info: Speaker information
            listener_info: Listener information

        Returns:
            SocialContextAnalysis: Social context analysis
        """
        # Calculate formality level
        formality_score = 0.0
        formality_count = 0

        for pattern, score in self.formality_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                formality_score += score * len(matches)
                formality_count += len(matches)

        if formality_count > 0:
            formality_level = min(formality_score / formality_count, 1.0)
        else:
            formality_level = 0.5  # Default neutral

        # Infer power distance
        power_distance = 0.0

        # Check for humble language (speaker lower)
        humble_indicators = ['드리다', '올리다', '여쭙다', '모시다']
        for indicator in humble_indicators:
            if indicator in text:
                power_distance -= 0.3

        # Check for elevated language (listener higher)
        elevated_indicators = ['께서', '님', '하십시오', '계시다']
        for indicator in elevated_indicators:
            if indicator in text:
                power_distance += 0.3

        power_distance = max(-1.0, min(1.0, power_distance))

        # Infer relationship type
        relationship = self._infer_relationship(text, context_hint, power_distance, formality_level)

        # Extract social markers
        social_markers = []
        for marker in self.honorific_markers.keys():
            if marker in text:
                social_marker = CulturalMarker(
                    text=marker,
                    marker_type='social',
                    cultural_significance=self.honorific_markers[marker]['type'],
                    start_pos=text.find(marker),
                    end_pos=text.find(marker) + len(marker),
                    confidence=self.honorific_markers[marker]['cultural_weight']
                )
                social_markers.append(social_marker)

        # Calculate confidence
        confidence = min((len(social_markers) * 0.2) + (formality_level * 0.5) + 0.3, 1.0)

        return SocialContextAnalysis(
            inferred_relationship=relationship,
            power_distance=power_distance,
            formality_level=formality_level,
            social_markers=social_markers,
            confidence=confidence
        )

    def _infer_relationship(self,
                          text: str,
                          context_hint: Optional[str],
                          power_distance: float,
                          formality_level: float) -> SocialRelationship:
        """Infer social relationship from linguistic cues."""

        # Check family terms
        family_terms = self.relationship_indicators['family_terms']
        for term in family_terms:
            if term in text:
                if term in ['아버지', '어머니', '할아버지', '할머니']:
                    return SocialRelationship.FAMILY_ELDER
                elif term in ['아들', '딸', '손자', '손녀', '동생']:
                    return SocialRelationship.FAMILY_YOUNGER

        # Check professional terms
        prof_terms = self.relationship_indicators['professional_titles']
        for term in prof_terms:
            if term in text:
                return SocialRelationship.PROFESSIONAL

        # Use context hint
        if context_hint:
            if context_hint in ['business', 'work', 'office']:
                return SocialRelationship.PROFESSIONAL
            elif context_hint in ['family', 'home']:
                if power_distance > 0:
                    return SocialRelationship.FAMILY_ELDER
                else:
                    return SocialRelationship.FAMILY_YOUNGER

        # Infer from power distance and formality
        if power_distance > 0.5 and formality_level > 0.7:
            return SocialRelationship.SOCIAL_SUPERIOR
        elif power_distance < -0.5:
            return SocialRelationship.SOCIAL_SUBORDINATE
        elif formality_level < 0.3:
            return SocialRelationship.PEER
        else:
            return SocialRelationship.STRANGER_OLDER

    async def _detect_cultural_concepts(self, text: str) -> List[str]:
        """
        Detect Korean cultural concepts in text.

        Args:
            text: Text to analyze

        Returns:
            List[str]: Detected cultural concepts
        """
        concepts = []

        for expression, info in self.cultural_expressions.items():
            if expression in text:
                concept = info.get('concept')
                if isinstance(concept, CulturalConcept):
                    concepts.append(concept.value)
                else:
                    concepts.append(str(concept))

        # Also check variants
        for expression, info in self.cultural_expressions.items():
            variants = info.get('variants', [])
            for variant in variants:
                if variant in text and info.get('concept'):
                    concept = info['concept']
                    if isinstance(concept, CulturalConcept):
                        concept_value = concept.value
                    else:
                        concept_value = str(concept)

                    if concept_value not in concepts:
                        concepts.append(concept_value)

        return concepts

    async def _extract_cultural_markers(self, text: str) -> List[CulturalMarker]:
        """
        Extract all cultural markers from text.

        Args:
            text: Text to analyze

        Returns:
            List[CulturalMarker]: Cultural markers
        """
        markers = []

        # Extract honorific markers
        for marker, info in self.honorific_markers.items():
            start = text.find(marker)
            if start != -1:
                cultural_marker = CulturalMarker(
                    text=marker,
                    marker_type='honorific',
                    cultural_significance=f"Honorific: {info['level'].value}",
                    start_pos=start,
                    end_pos=start + len(marker),
                    confidence=info['cultural_weight']
                )
                markers.append(cultural_marker)

        # Extract cultural expression markers
        for expression, info in self.cultural_expressions.items():
            start = text.find(expression)
            if start != -1:
                cultural_marker = CulturalMarker(
                    text=expression,
                    marker_type='cultural_expression',
                    cultural_significance=info.get('cultural_note', 'Cultural expression'),
                    start_pos=start,
                    end_pos=start + len(expression),
                    confidence=0.8
                )
                markers.append(cultural_marker)

        return markers

    async def _generate_appropriateness_suggestions(self,
                                                  text: str,
                                                  honorific_analysis: HonorificAnalysis,
                                                  social_context: SocialContextAnalysis) -> List[str]:
        """
        Generate suggestions for more appropriate language use.

        Args:
            text: Original text
            honorific_analysis: Honorific analysis
            social_context: Social context analysis

        Returns:
            List[str]: Appropriateness suggestions
        """
        suggestions = []

        # Check honorific consistency
        if honorific_analysis.consistency_score < 0.7:
            suggestions.append(
                f"Consider using consistent honorific level throughout. "
                f"Current level: {honorific_analysis.overall_level.value}"
            )

        # Check appropriateness for relationship
        relationship = social_context.inferred_relationship

        if relationship == SocialRelationship.PROFESSIONAL:
            if honorific_analysis.overall_level in [HonorificLevel.CASUAL, HonorificLevel.POLITE_INFORMAL]:
                suggestions.append(
                    "For professional context, consider using more formal language "
                    "such as '습니다' endings"
                )

        elif relationship == SocialRelationship.FAMILY_ELDER:
            if honorific_analysis.overall_level not in [HonorificLevel.ELEVATED, HonorificLevel.POLITE_FORMAL]:
                suggestions.append(
                    "When addressing family elders, consider using elevated honorifics "
                    "like '하십니다' or '께서'"
                )

        elif relationship == SocialRelationship.PEER:
            if honorific_analysis.overall_level == HonorificLevel.SUPER_ELEVATED:
                suggestions.append(
                    "For peer conversation, the language might be overly formal. "
                    "Consider using '해요' style"
                )

        # Check cultural appropriateness
        if social_context.formality_level < 0.3 and social_context.power_distance > 0.5:
            suggestions.append(
                "The language appears too casual for the hierarchical context. "
                "Consider increasing formality level"
            )

        # Suggest cultural concepts usage
        if not any(concept in ['nunchi', 'jeong', 'uri'] for concept in []):
            # This is a placeholder - in practice, you'd have more sophisticated logic
            pass

        return suggestions[:5]  # Limit to top 5 suggestions

    def _calculate_cultural_confidence(self,
                                     honorific_analysis: HonorificAnalysis,
                                     social_context: SocialContextAnalysis,
                                     cultural_concepts: List[str],
                                     cultural_markers: List[CulturalMarker]) -> float:
        """
        Calculate overall confidence for cultural analysis.

        Args:
            honorific_analysis: Honorific analysis
            social_context: Social context analysis
            cultural_concepts: Detected concepts
            cultural_markers: Cultural markers

        Returns:
            float: Overall confidence score
        """
        confidences = []

        # Honorific analysis confidence
        honorific_confidence = (
            honorific_analysis.consistency_score * 0.5 +
            honorific_analysis.appropriateness_score * 0.3 +
            min(len(honorific_analysis.specific_markers) * 0.1, 0.2)
        )
        confidences.append(honorific_confidence)

        # Social context confidence
        confidences.append(social_context.confidence)

        # Cultural concepts confidence
        concept_confidence = min(len(cultural_concepts) * 0.2, 1.0)
        confidences.append(concept_confidence)

        # Cultural markers confidence
        if cultural_markers:
            marker_confidence = sum(m.confidence for m in cultural_markers) / len(cultural_markers)
            confidences.append(marker_confidence)

        return sum(confidences) / len(confidences)

    async def get_cultural_statistics(self, result: CulturalContextResult) -> Dict[str, Any]:
        """
        Get statistics for cultural analysis result.

        Args:
            result: Cultural context result

        Returns:
            Dict[str, Any]: Cultural analysis statistics
        """
        stats = {
            'text_length': len(result.text),
            'overall_confidence': result.overall_confidence,
            'honorific_analysis': {
                'level': result.honorific_analysis.overall_level.value,
                'consistency': result.honorific_analysis.consistency_score,
                'appropriateness': result.honorific_analysis.appropriateness_score,
                'marker_count': len(result.honorific_analysis.specific_markers)
            },
            'social_context': {
                'relationship': result.social_context.inferred_relationship.value,
                'power_distance': result.social_context.power_distance,
                'formality_level': result.social_context.formality_level,
                'confidence': result.social_context.confidence
            },
            'cultural_concepts': {
                'count': len(result.cultural_concepts),
                'concepts': result.cultural_concepts
            },
            'cultural_markers': {
                'count': len(result.cultural_markers),
                'by_type': {}
            },
            'suggestions_count': len(result.appropriateness_suggestions)
        }

        # Count markers by type
        for marker in result.cultural_markers:
            marker_type = marker.marker_type
            stats['cultural_markers']['by_type'][marker_type] = \
                stats['cultural_markers']['by_type'].get(marker_type, 0) + 1

        return stats