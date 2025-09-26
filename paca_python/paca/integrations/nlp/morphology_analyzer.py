"""
Module: integrations.nlp.morphology_analyzer
Purpose: Advanced Korean morphological analysis with linguistic features
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

from .konlpy_integration import KoNLPyIntegration, MorphemeResult

logger = logging.getLogger(__name__)

class InflectionType(Enum):
    """Korean word inflection types."""
    VERB_CONJUGATION = "verb_conjugation"
    ADJECTIVE_CONJUGATION = "adjective_conjugation"
    NOUN_DECLENSION = "noun_declension"
    IRREGULAR = "irregular"
    COMPOUND = "compound"

class WordClass(Enum):
    """Korean word classes with detailed categories."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PARTICLE = "particle"
    ENDING = "ending"
    MODIFIER = "modifier"
    INTERJECTION = "interjection"
    DETERMINER = "determiner"

@dataclass
class MorphologyResult:
    """Comprehensive morphological analysis result."""
    surface_form: str
    lemma: str
    pos_tag: str
    word_class: WordClass
    features: Dict[str, Any]
    inflection_type: Optional[InflectionType] = None
    is_compound: bool = False
    compound_parts: List[str] = None
    confidence: float = 1.0
    phonetic_form: Optional[str] = None

@dataclass
class AnalysisContext:
    """Context information for morphological analysis."""
    sentence: str
    word_position: int
    neighboring_words: List[str]
    honorific_context: Optional[str] = None

class MorphologyAnalyzer:
    """
    Advanced Korean morphological analyzer with comprehensive linguistic analysis.

    Provides detailed morphological analysis including inflection detection,
    compound word analysis, phonetic processing, and contextual features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize morphology analyzer.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.nlp_integration = KoNLPyIntegration()

        # Load linguistic resources
        self.irregular_verbs = self._load_irregular_verbs()
        self.compound_patterns = self._load_compound_patterns()
        self.phonetic_rules = self._load_phonetic_rules()
        self.inflection_patterns = self._load_inflection_patterns()

        # POS tag mappings
        self.pos_mappings = self._create_pos_mappings()

        self._initialized = False

    def _load_irregular_verbs(self) -> Dict[str, Dict[str, Any]]:
        """Load irregular verb patterns and their analysis rules."""
        return {
            # ㅅ 불규칙 (ㅅ irregular)
            '짓다': {'stem': '짓', 'type': 'ㅅ_irregular', 'forms': ['지어', '지으']},
            '잇다': {'stem': '잇', 'type': 'ㅅ_irregular', 'forms': ['이어', '이으']},

            # ㄷ 불규칙 (ㄷ irregular)
            '걷다': {'stem': '걷', 'type': 'ㄷ_irregular', 'forms': ['걸어', '걸으']},
            '듣다': {'stem': '듣', 'type': 'ㄷ_irregular', 'forms': ['들어', '들으']},

            # ㅂ 불규칙 (ㅂ irregular)
            '굽다': {'stem': '굽', 'type': 'ㅂ_irregular', 'forms': ['구워', '구우']},
            '춥다': {'stem': '춥', 'type': 'ㅂ_irregular', 'forms': ['추워', '추우']},

            # ㅎ 불규칙 (ㅎ irregular)
            '그렇다': {'stem': '그렇', 'type': 'ㅎ_irregular', 'forms': ['그래', '그러']},
            '이렇다': {'stem': '이렇', 'type': 'ㅎ_irregular', 'forms': ['이래', '이러']},

            # 르 불규칙 (르 irregular)
            '부르다': {'stem': '부르', 'type': '르_irregular', 'forms': ['불러']},
            '고르다': {'stem': '고르', 'type': '르_irregular', 'forms': ['골라']},

            # ㄹ 탈락 (ㄹ deletion)
            '만들다': {'stem': '만들', 'type': 'ㄹ_deletion', 'forms': ['만들어', '만드니']},
            '살다': {'stem': '살', 'type': 'ㄹ_deletion', 'forms': ['살아', '사니']},
        }

    def _load_compound_patterns(self) -> Dict[str, List[str]]:
        """Load compound word patterns."""
        return {
            'noun_noun': ['+', '의', ''],  # 학교+건물, 학교의건물, 학교건물
            'verb_noun': ['기', '는것', '음'],  # 읽기, 읽는것, 읽음
            'adjective_noun': ['함', '기', '음'],  # 아름다움, 좋기, 큼
            'modifier_patterns': ['한', '을', '는', '던']
        }

    def _load_phonetic_rules(self) -> Dict[str, List[Dict[str, str]]]:
        """Load Korean phonetic transformation rules."""
        return {
            'liaison': [
                {
                    'pattern': r'([ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ])\s*([ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ])',
                    'replacement': r'\1\2'
                },
            ],
            'assimilation': [
                {'pattern': r'ㄱ+ㄴ', 'replacement': 'ㅇㄴ'},  # 막내 -> 망내
                {'pattern': r'ㄷ+ㄴ', 'replacement': 'ㄴㄴ'},  # 듣는 -> 든는
            ],
            'fortition': [
                {
                    'pattern': r'([ㄱㄷㅂ])\s*([ㄱㄷㅂㅅㅈ])',
                    'replacement': r'\1\2'
                },
            ]
        }

    def _load_inflection_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load inflection patterns for verbs and adjectives."""
        return {
            'verb_endings': {
                'present': ['다', '는다', '습니다', '해', '해요'],
                'past': ['았다', '었다', '했다', '았어', '었어', '했어'],
                'future': ['겠다', '을거다', 'ㄹ거다', '겠어', '을거야'],
                'interrogative': ['나', '니', '냐', '까', '니까'],
                'imperative': ['라', '어라', '아라', '세요', '십시오'],
                'propositive': ['자', '아요', '어요', '해요']
            },
            'adjective_endings': {
                'attributive': ['은', '는', '을', '는', 'ㄴ'],
                'predicative': ['다', '습니다', '해', '해요'],
                'adverbial': ['게', '히', '이', '으로']
            },
            'honorific_endings': {
                'formal': ['습니다', '습니까', '십니다', '십니까'],
                'informal_polite': ['해요', '해', '어요', '아요'],
                'humble': ['드립니다', '올립니다', '여쭙니다']
            }
        }

    def _create_pos_mappings(self) -> Dict[str, WordClass]:
        """Create POS tag to word class mappings."""
        return {
            # Nouns
            'NNG': WordClass.NOUN,    # 일반명사
            'NNP': WordClass.NOUN,    # 고유명사
            'NNB': WordClass.NOUN,    # 의존명사
            'NR': WordClass.NOUN,     # 수사
            'NP': WordClass.NOUN,     # 대명사

            # Verbs
            'VV': WordClass.VERB,     # 동사
            'VA': WordClass.ADJECTIVE, # 형용사
            'VX': WordClass.VERB,     # 보조동사
            'VCP': WordClass.VERB,    # 긍정지정사
            'VCN': WordClass.VERB,    # 부정지정사

            # Modifiers
            'MM': WordClass.MODIFIER, # 관형사
            'MAG': WordClass.ADVERB,  # 일반부사
            'MAJ': WordClass.ADVERB,  # 접속부사

            # Particles
            'JKS': WordClass.PARTICLE, # 주격조사
            'JKC': WordClass.PARTICLE, # 보격조사
            'JKG': WordClass.PARTICLE, # 관형격조사
            'JKO': WordClass.PARTICLE, # 목적격조사
            'JKB': WordClass.PARTICLE, # 부사격조사
            'JKV': WordClass.PARTICLE, # 호격조사
            'JKQ': WordClass.PARTICLE, # 인용격조사
            'JX': WordClass.PARTICLE,  # 보조사
            'JC': WordClass.PARTICLE,  # 접속조사

            # Endings
            'EP': WordClass.ENDING,   # 선어말어미
            'EF': WordClass.ENDING,   # 종결어미
            'EC': WordClass.ENDING,   # 연결어미
            'ETN': WordClass.ENDING,  # 명사형전성어미
            'ETM': WordClass.ENDING,  # 관형사형전성어미

            # Others
            'IC': WordClass.INTERJECTION, # 감탄사
        }

    async def initialize(self) -> bool:
        """
        Initialize the morphology analyzer.

        Returns:
            bool: True if initialization successful
        """
        try:
            await self.nlp_integration.initialize()
            self._initialized = True
            logger.info("Morphology analyzer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize morphology analyzer: {e}")
            return False

    async def analyze(self,
                     text: str,
                     context: Optional[AnalysisContext] = None,
                     include_phonetics: bool = False,
                     include_compounds: bool = True,
                     include_inflections: bool = True) -> List[MorphologyResult]:
        """
        Perform comprehensive morphological analysis.

        Args:
            text: Korean text to analyze
            context: Analysis context information
            include_phonetics: Include phonetic analysis
            include_compounds: Analyze compound words
            include_inflections: Analyze inflections

        Returns:
            List[MorphologyResult]: Comprehensive analysis results
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Get basic morphological analysis
            basic_morphemes = await self.nlp_integration.analyze_morphology(text)

            results = []

            for morpheme in basic_morphemes:
                # Create comprehensive analysis
                result = await self._create_comprehensive_analysis(
                    morpheme, context, include_phonetics, include_compounds, include_inflections
                )
                results.append(result)

            processing_time = time.time() - start_time
            logger.debug(f"Morphological analysis completed: {len(results)} morphemes in {processing_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"Morphological analysis failed: {e}")
            raise

    async def _create_comprehensive_analysis(self,
                                           morpheme: MorphemeResult,
                                           context: Optional[AnalysisContext],
                                           include_phonetics: bool,
                                           include_compounds: bool,
                                           include_inflections: bool) -> MorphologyResult:
        """
        Create comprehensive morphological analysis for a single morpheme.

        Args:
            morpheme: Basic morpheme result
            context: Analysis context
            include_phonetics: Include phonetic analysis
            include_compounds: Analyze compounds
            include_inflections: Analyze inflections

        Returns:
            MorphologyResult: Comprehensive analysis
        """
        # Determine word class
        word_class = self.pos_mappings.get(morpheme.pos, WordClass.NOUN)

        # Initialize result
        result = MorphologyResult(
            surface_form=morpheme.surface,
            lemma=morpheme.lemma,
            pos_tag=morpheme.pos,
            word_class=word_class,
            features=morpheme.features.copy(),
            confidence=morpheme.confidence
        )

        # Analyze inflections
        if include_inflections:
            inflection_info = await self._analyze_inflection(morpheme, word_class)
            result.inflection_type = inflection_info.get('type')
            result.features.update(inflection_info.get('features', {}))

        # Analyze compounds
        if include_compounds:
            compound_info = await self._analyze_compound(morpheme.surface)
            result.is_compound = compound_info['is_compound']
            result.compound_parts = compound_info.get('parts', [])

        # Generate phonetic form
        if include_phonetics:
            result.phonetic_form = await self._generate_phonetic_form(morpheme.surface)

        # Add contextual features
        if context:
            contextual_features = await self._analyze_context(morpheme, context)
            result.features.update(contextual_features)

        return result

    async def _analyze_inflection(self,
                                morpheme: MorphemeResult,
                                word_class: WordClass) -> Dict[str, Any]:
        """
        Analyze inflection patterns for the morpheme.

        Args:
            morpheme: Morpheme to analyze
            word_class: Word class of the morpheme

        Returns:
            Dict[str, Any]: Inflection analysis
        """
        surface = morpheme.surface
        pos = morpheme.pos

        inflection_info = {
            'type': None,
            'features': {}
        }

        # Check for irregular verbs
        if surface in self.irregular_verbs:
            irregular_data = self.irregular_verbs[surface]
            inflection_info['type'] = InflectionType.IRREGULAR
            inflection_info['features'].update({
                'irregular_type': irregular_data['type'],
                'stem': irregular_data['stem'],
                'irregular_forms': irregular_data['forms']
            })

        # Analyze verb conjugations
        elif word_class == WordClass.VERB:
            verb_features = await self._analyze_verb_conjugation(surface, pos)
            if verb_features:
                inflection_info['type'] = InflectionType.VERB_CONJUGATION
                inflection_info['features'].update(verb_features)

        # Analyze adjective conjugations
        elif word_class == WordClass.ADJECTIVE:
            adj_features = await self._analyze_adjective_conjugation(surface, pos)
            if adj_features:
                inflection_info['type'] = InflectionType.ADJECTIVE_CONJUGATION
                inflection_info['features'].update(adj_features)

        return inflection_info

    async def _analyze_verb_conjugation(self, surface: str, pos: str) -> Dict[str, Any]:
        """Analyze verb conjugation patterns."""
        features = {}

        verb_endings = self.inflection_patterns['verb_endings']

        # Check tense
        for tense, endings in verb_endings.items():
            for ending in endings:
                if surface.endswith(ending):
                    features['tense'] = tense
                    features['ending'] = ending
                    break

        # Check honorific forms
        honorific_endings = self.inflection_patterns['honorific_endings']
        for level, endings in honorific_endings.items():
            for ending in endings:
                if surface.endswith(ending):
                    features['honorific_level'] = level
                    features['honorific_ending'] = ending
                    break

        return features

    async def _analyze_adjective_conjugation(self, surface: str, pos: str) -> Dict[str, Any]:
        """Analyze adjective conjugation patterns."""
        features = {}

        adj_endings = self.inflection_patterns['adjective_endings']

        # Check adjective forms
        for form, endings in adj_endings.items():
            for ending in endings:
                if surface.endswith(ending):
                    features['adjective_form'] = form
                    features['ending'] = ending
                    break

        return features

    async def _analyze_compound(self, surface: str) -> Dict[str, Any]:
        """
        Analyze compound word structure.

        Args:
            surface: Surface form to analyze

        Returns:
            Dict[str, Any]: Compound analysis
        """
        compound_info = {
            'is_compound': False,
            'parts': []
        }

        # Simple heuristic: check for common compound patterns
        # This is a simplified implementation - real compound analysis is more complex

        if len(surface) > 4:  # Minimum length for compound
            # Try to split based on common patterns
            possible_splits = []

            # Check for noun+noun compounds
            for i in range(2, len(surface) - 1):
                part1 = surface[:i]
                part2 = surface[i:]

                # Simple check if both parts could be valid morphemes
                if len(part1) >= 2 and len(part2) >= 2:
                    possible_splits.append([part1, part2])

            if possible_splits:
                # Take the most balanced split as the best candidate
                best_split = min(possible_splits, key=lambda x: abs(len(x[0]) - len(x[1])))
                compound_info['is_compound'] = True
                compound_info['parts'] = best_split

        return compound_info

    async def _generate_phonetic_form(self, surface: str) -> str:
        """
        Generate phonetic representation of Korean text.

        Args:
            surface: Surface form

        Returns:
            str: Phonetic form
        """
        # This is a simplified phonetic transformation
        # Real Korean phonetics involves complex rules

        phonetic = surface

        # Apply basic phonetic rules
        for rule_type, rules in self.phonetic_rules.items():
            for rule in rules:
                pattern = rule['pattern']
                replacement = rule['replacement']
                phonetic = re.sub(pattern, replacement, phonetic)

        return phonetic

    async def _analyze_context(self,
                             morpheme: MorphemeResult,
                             context: AnalysisContext) -> Dict[str, Any]:
        """
        Analyze contextual features of the morpheme.

        Args:
            morpheme: Morpheme to analyze
            context: Analysis context

        Returns:
            Dict[str, Any]: Contextual features
        """
        features = {}

        # Position in sentence
        features['sentence_position'] = context.word_position

        # Neighboring word influence
        if context.neighboring_words:
            features['has_neighbors'] = True
            features['neighbor_count'] = len(context.neighboring_words)

            # Check for specific patterns with neighbors
            if context.word_position > 0:
                prev_word = context.neighboring_words[context.word_position - 1]
                features['previous_word'] = prev_word

            if context.word_position < len(context.neighboring_words) - 1:
                next_word = context.neighboring_words[context.word_position + 1]
                features['next_word'] = next_word

        # Honorific context
        if context.honorific_context:
            features['sentence_honorific_level'] = context.honorific_context

        return features

    async def get_lemma(self, surface: str, pos: str) -> str:
        """
        Get the lemma (dictionary form) for a given surface form.

        Args:
            surface: Surface form
            pos: Part-of-speech tag

        Returns:
            str: Lemma form
        """
        # For verbs and adjectives, try to get the dictionary form
        if pos.startswith('V'):  # Verb or adjective
            # Check if it's an irregular verb
            if surface in self.irregular_verbs:
                return self.irregular_verbs[surface]['stem'] + '다'

            # Try to extract stem and add 다
            if surface.endswith(('어', '아', '여', '해')):
                # Remove ending and add 다
                stem = surface[:-1] if len(surface) > 1 else surface
                return stem + '다'

        # For other word classes, return as-is
        return surface

    async def get_analysis_statistics(self, results: List[MorphologyResult]) -> Dict[str, Any]:
        """
        Get statistics for morphological analysis results.

        Args:
            results: Analysis results

        Returns:
            Dict[str, Any]: Statistics
        """
        if not results:
            return {}

        stats = {
            'total_morphemes': len(results),
            'word_classes': {},
            'inflection_types': {},
            'compound_ratio': 0,
            'irregular_ratio': 0,
            'average_confidence': 0
        }

        compound_count = 0
        irregular_count = 0
        total_confidence = 0

        for result in results:
            # Count word classes
            wc = result.word_class.value
            stats['word_classes'][wc] = stats['word_classes'].get(wc, 0) + 1

            # Count inflection types
            if result.inflection_type:
                it = result.inflection_type.value
                stats['inflection_types'][it] = stats['inflection_types'].get(it, 0) + 1

            # Count compounds and irregulars
            if result.is_compound:
                compound_count += 1

            if result.inflection_type == InflectionType.IRREGULAR:
                irregular_count += 1

            total_confidence += result.confidence

        # Calculate ratios
        stats['compound_ratio'] = compound_count / len(results)
        stats['irregular_ratio'] = irregular_count / len(results)
        stats['average_confidence'] = total_confidence / len(results)

        return stats
