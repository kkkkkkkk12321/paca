"""
Module: integrations.nlp.korean_tokenizer
Purpose: Korean text tokenization with enhanced features for honorifics and context
Author: PACA Development Team
Created: 2024-09-24
Last Modified: 2024-09-24
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time

from .konlpy_integration import KoNLPyIntegration, AnalyzerType

logger = logging.getLogger(__name__)

class TokenType(Enum):
    """Types of tokens in Korean text."""
    WORD = "word"
    HONORIFIC = "honorific"
    PARTICLE = "particle"
    ENDING = "ending"
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    FOREIGN = "foreign"
    EMOJI = "emoji"
    WHITESPACE = "whitespace"

class HonorificLevel(Enum):
    """Korean honorific levels."""
    CASUAL = "casual"          # 반말
    POLITE = "polite"         # 존댓말 (해요체)
    FORMAL = "formal"         # 격식체 (합니다체)
    HUMBLE = "humble"         # 겸양어
    ELEVATED = "elevated"     # 높임말

@dataclass
class Token:
    """Korean token with enhanced information."""
    text: str
    token_type: TokenType
    start_pos: int
    end_pos: int
    morphemes: List[Dict[str, Any]]
    honorific_level: Optional[HonorificLevel] = None
    confidence: float = 1.0
    features: Optional[Dict[str, Any]] = None

class KoreanTokenizer:
    """
    Advanced Korean tokenizer with honorific detection and cultural context.

    Provides sophisticated tokenization capabilities specifically designed
    for Korean language processing, including honorific level detection,
    cultural context analysis, and morphological decomposition.
    """

    def __init__(self,
                 model_type: str = "mecab",
                 include_honorifics: bool = True,
                 include_morphology: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Korean tokenizer.

        Args:
            model_type: Underlying morphological analyzer type
            include_honorifics: Enable honorific detection
            include_morphology: Include morphological analysis
            config: Additional configuration parameters
        """
        self.model_type = model_type
        self.include_honorifics = include_honorifics
        self.include_morphology = include_morphology
        self.config = config or {}

        # Initialize KoNLPy integration
        analyzer_type = AnalyzerType(model_type.lower())
        self.nlp_integration = KoNLPyIntegration(primary_analyzer=analyzer_type)

        # Honorific patterns
        self.honorific_patterns = self._load_honorific_patterns()
        self.formal_endings = self._load_formal_endings()
        self.humble_words = self._load_humble_words()

        # Special token patterns
        self.special_patterns = self._compile_special_patterns()

        self._initialized = False

    def _load_honorific_patterns(self) -> Dict[str, List[str]]:
        """Load Korean honorific patterns."""
        return {
            'formal_endings': [
                '습니다', '습니까', '겠습니다', '겠습니까',
                '하십니다', '하십니까', '이십니다', '이십니까'
            ],
            'polite_endings': [
                '해요', '해', '요', '죠', '예요', '이에요',
                '해요?', '해?', '요?', '죠?'
            ],
            'humble_prefixes': ['드', '올', '여쭈', '여쭙'],
            'elevated_prefixes': ['주', '께서', '시'],
            'honorific_titles': [
                '님', '씨', '군', '양', '선생님', '교수님',
                '박사님', '회장님', '사장님', '부장님'
            ]
        }

    def _load_formal_endings(self) -> Set[str]:
        """Load formal ending patterns."""
        return {
            '다', '까', '냐', '라', '자', '지', '네', '구나',
            '습니다', '습니까', '겠습니다', '겠습니까'
        }

    def _load_humble_words(self) -> Set[str]:
        """Load humble word patterns."""
        return {
            '드리다', '올리다', '여쭙다', '여쭈다', '모시다',
            '뵙다', '뵈다', '받들다', '섬기다'
        }

    def _compile_special_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for special tokens."""
        return {
            'number': re.compile(r'\d+'),
            'emoji': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'),
            'punctuation': re.compile(r'[.,!?;:\-(){}[\]"\'`~@#$%^&*+=<>/\\|_]'),
            'whitespace': re.compile(r'\s+'),
            'foreign': re.compile(r'[a-zA-Z]+'),
            'korean_char': re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]+')
        }

    async def initialize(self) -> bool:
        """
        Initialize the tokenizer and underlying NLP components.

        Returns:
            bool: True if initialization successful
        """
        try:
            await self.nlp_integration.initialize()
            self._initialized = True
            logger.info(f"Korean tokenizer initialized with {self.model_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Korean tokenizer: {e}")
            return False

    async def tokenize(self, text: str,
                      include_pos: bool = True,
                      include_honorifics: bool = None,
                      preserve_whitespace: bool = False) -> List[Token]:
        """
        Tokenize Korean text with enhanced features.

        Args:
            text: Input Korean text
            include_pos: Include part-of-speech information
            include_honorifics: Detect honorific levels
            preserve_whitespace: Keep whitespace tokens

        Returns:
            List[Token]: List of tokenized elements
        """
        if not self._initialized:
            await self.initialize()

        if include_honorifics is None:
            include_honorifics = self.include_honorifics

        start_time = time.time()

        try:
            tokens = []
            current_pos = 0

            # First pass: identify special tokens
            special_tokens = await self._identify_special_tokens(text)

            # Second pass: morphological analysis for Korean text
            if self.include_morphology:
                morphology_results = await self.nlp_integration.analyze_morphology(text)
            else:
                morphology_results = []

            # Third pass: combine and create enhanced tokens
            tokens = await self._create_enhanced_tokens(
                text, special_tokens, morphology_results,
                include_pos, include_honorifics, preserve_whitespace
            )

            # Fourth pass: detect honorific context
            if include_honorifics:
                await self._detect_honorific_context(tokens, text)

            processing_time = time.time() - start_time
            logger.debug(f"Tokenization completed: {len(tokens)} tokens in {processing_time:.3f}s")

            return tokens

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback to simple tokenization
            return await self._simple_tokenize(text, preserve_whitespace)

    async def _identify_special_tokens(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify special tokens (numbers, punctuation, emoji, etc.).

        Args:
            text: Input text

        Returns:
            List[Dict[str, Any]]: Special tokens with positions
        """
        special_tokens = []

        # Find all special patterns
        for token_type, pattern in self.special_patterns.items():
            for match in pattern.finditer(text):
                special_tokens.append({
                    'type': token_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })

        # Sort by position
        special_tokens.sort(key=lambda x: x['start'])

        return special_tokens

    async def _create_enhanced_tokens(self,
                                    text: str,
                                    special_tokens: List[Dict[str, Any]],
                                    morphology_results: List[Any],
                                    include_pos: bool,
                                    include_honorifics: bool,
                                    preserve_whitespace: bool) -> List[Token]:
        """
        Create enhanced tokens combining special tokens and morphology.

        Args:
            text: Original text
            special_tokens: Special token information
            morphology_results: Morphological analysis results
            include_pos: Include POS tags
            include_honorifics: Include honorific detection
            preserve_whitespace: Keep whitespace tokens

        Returns:
            List[Token]: Enhanced tokens
        """
        tokens = []
        current_pos = 0

        # Create morpheme lookup for position mapping
        morpheme_lookup = {}
        morpheme_pos = 0

        for morpheme in morphology_results:
            morpheme_lookup[morpheme_pos] = morpheme
            morpheme_pos += len(morpheme.surface)

        # Process text character by character
        i = 0
        while i < len(text):
            # Check for special tokens
            special_token = self._get_special_token_at_position(special_tokens, i)

            if special_token:
                # Create special token
                token = Token(
                    text=special_token['text'],
                    token_type=TokenType(special_token['type']),
                    start_pos=special_token['start'],
                    end_pos=special_token['end'],
                    morphemes=[],
                    features={'is_special': True}
                )

                if token.token_type != TokenType.WHITESPACE or preserve_whitespace:
                    tokens.append(token)

                i = special_token['end']

            else:
                # Find corresponding morpheme
                morpheme = self._get_morpheme_at_position(morpheme_lookup, i)

                if morpheme:
                    # Create morphological token
                    token_type = self._determine_token_type(morpheme)
                    honorific_level = None

                    if include_honorifics:
                        honorific_level = self._detect_token_honorific(morpheme)

                    morpheme_data = [morpheme.__dict__] if include_pos else []

                    token = Token(
                        text=morpheme.surface,
                        token_type=token_type,
                        start_pos=i,
                        end_pos=i + len(morpheme.surface),
                        morphemes=morpheme_data,
                        honorific_level=honorific_level,
                        confidence=morpheme.confidence,
                        features={
                            'pos': morpheme.pos,
                            'pos_category': morpheme.features.get('pos_category', 'unknown')
                        }
                    )

                    tokens.append(token)
                    i += len(morpheme.surface)

                else:
                    # Skip unrecognized character
                    i += 1

        return tokens

    def _get_special_token_at_position(self,
                                     special_tokens: List[Dict[str, Any]],
                                     position: int) -> Optional[Dict[str, Any]]:
        """Get special token at specific position."""
        for token in special_tokens:
            if token['start'] <= position < token['end']:
                return token
        return None

    def _get_morpheme_at_position(self,
                                morpheme_lookup: Dict[int, Any],
                                position: int) -> Optional[Any]:
        """Get morpheme at specific position."""
        return morpheme_lookup.get(position)

    def _determine_token_type(self, morpheme: Any) -> TokenType:
        """Determine token type from morpheme."""
        pos = morpheme.pos

        if pos.startswith('N'):  # Noun
            return TokenType.WORD
        elif pos.startswith('J'):  # Particle
            return TokenType.PARTICLE
        elif pos.startswith('E'):  # Ending
            return TokenType.ENDING
        else:
            return TokenType.WORD

    def _detect_token_honorific(self, morpheme: Any) -> Optional[HonorificLevel]:
        """Detect honorific level for individual token."""
        surface = morpheme.surface.lower()
        pos = morpheme.pos

        # Check for formal endings
        for formal_ending in self.honorific_patterns['formal_endings']:
            if surface.endswith(formal_ending):
                return HonorificLevel.FORMAL

        # Check for polite endings
        for polite_ending in self.honorific_patterns['polite_endings']:
            if surface.endswith(polite_ending):
                return HonorificLevel.POLITE

        # Check for humble words
        if surface in self.humble_words:
            return HonorificLevel.HUMBLE

        # Check for honorific titles
        for title in self.honorific_patterns['honorific_titles']:
            if title in surface:
                return HonorificLevel.ELEVATED

        return None

    async def _detect_honorific_context(self, tokens: List[Token], text: str) -> None:
        """Detect overall honorific context of the text."""
        honorific_counts = {level: 0 for level in HonorificLevel}

        # Count honorific markers
        for token in tokens:
            if token.honorific_level:
                honorific_counts[token.honorific_level] += 1

        # Determine dominant honorific level
        dominant_level = max(honorific_counts.items(), key=lambda x: x[1])

        # Apply context to tokens without explicit honorific markers
        for token in tokens:
            if not token.honorific_level and dominant_level[1] > 0:
                if token.token_type in [TokenType.WORD, TokenType.ENDING]:
                    # Apply contextual honorific level with lower confidence
                    token.honorific_level = dominant_level[0]
                    token.confidence *= 0.7

    async def _simple_tokenize(self, text: str, preserve_whitespace: bool) -> List[Token]:
        """
        Fallback simple tokenization when advanced methods fail.

        Args:
            text: Input text
            preserve_whitespace: Keep whitespace tokens

        Returns:
            List[Token]: Simple tokens
        """
        tokens = []
        words = text.split()
        current_pos = 0

        for word in words:
            # Find actual position in original text
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)

            token = Token(
                text=word,
                token_type=TokenType.WORD,
                start_pos=word_start,
                end_pos=word_end,
                morphemes=[],
                features={'fallback': True}
            )

            tokens.append(token)
            current_pos = word_end

        logger.warning("Used fallback tokenization due to processing error")
        return tokens

    async def detect_honorifics(self, text: str) -> Dict[str, Any]:
        """
        Detect honorific level and patterns in Korean text.

        Args:
            text: Korean text to analyze

        Returns:
            Dict[str, Any]: Honorific analysis results
        """
        if not self._initialized:
            await self.initialize()

        try:
            tokens = await self.tokenize(text, include_honorifics=True)

            # Count honorific patterns
            honorific_counts = {level.value: 0 for level in HonorificLevel}
            total_tokens = len([t for t in tokens if t.token_type not in [TokenType.PUNCTUATION, TokenType.WHITESPACE]])

            for token in tokens:
                if token.honorific_level:
                    honorific_counts[token.honorific_level.value] += 1

            # Determine dominant level
            if total_tokens > 0:
                honorific_percentages = {
                    level: (count / total_tokens) * 100
                    for level, count in honorific_counts.items()
                }
            else:
                honorific_percentages = {level: 0 for level in honorific_counts}

            dominant_level = max(honorific_counts.items(), key=lambda x: x[1])

            return {
                'dominant_level': dominant_level[0] if dominant_level[1] > 0 else 'neutral',
                'honorific_counts': honorific_counts,
                'honorific_percentages': honorific_percentages,
                'total_tokens': total_tokens,
                'confidence': dominant_level[1] / max(total_tokens, 1)
            }

        except Exception as e:
            logger.error(f"Honorific detection failed: {e}")
            return {
                'dominant_level': 'unknown',
                'error': str(e)
            }

    async def get_statistics(self, tokens: List[Token]) -> Dict[str, Any]:
        """
        Get detailed statistics for tokenized text.

        Args:
            tokens: List of tokens

        Returns:
            Dict[str, Any]: Token statistics
        """
        stats = {
            'total_tokens': len(tokens),
            'token_types': {},
            'honorific_levels': {},
            'pos_tags': {},
            'average_token_length': 0,
            'korean_ratio': 0
        }

        if not tokens:
            return stats

        # Count by token type
        for token in tokens:
            token_type = token.token_type.value
            stats['token_types'][token_type] = stats['token_types'].get(token_type, 0) + 1

            # Count honorific levels
            if token.honorific_level:
                level = token.honorific_level.value
                stats['honorific_levels'][level] = stats['honorific_levels'].get(level, 0) + 1

            # Count POS tags
            for morpheme in token.morphemes:
                pos = morpheme.get('pos', 'unknown')
                stats['pos_tags'][pos] = stats['pos_tags'].get(pos, 0) + 1

        # Calculate averages
        total_length = sum(len(token.text) for token in tokens)
        stats['average_token_length'] = total_length / len(tokens)

        # Calculate Korean text ratio
        korean_tokens = [t for t in tokens if self.special_patterns['korean_char'].match(t.text)]
        stats['korean_ratio'] = len(korean_tokens) / len(tokens)

        return stats