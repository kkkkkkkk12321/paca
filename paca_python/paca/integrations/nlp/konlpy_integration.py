"""
Module: integrations.nlp.konlpy_integration
Purpose: KoNLPy library integration for Korean morphological analysis
Author: PACA Development Team
Created: 2024-09-24
Last Modified: 2024-09-24
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class AnalyzerType(Enum):
    """Supported KoNLPy analyzers."""
    MECAB = "mecab"
    HANNANUM = "hannanum"
    KKMA = "kkma"
    KOMORAN = "komoran"
    OKT = "okt"  # Open Korean Text (formerly Twitter)

@dataclass
class MorphemeResult:
    """Single morpheme analysis result."""
    surface: str  # 어절
    lemma: str    # 기본형
    pos: str      # 품사
    features: Dict[str, Any]  # 추가 특성
    confidence: float = 1.0

@dataclass
class TokenResult:
    """Token analysis result."""
    text: str
    start_pos: int
    end_pos: int
    morphemes: List[MorphemeResult]

class KoNLPyIntegration:
    """
    KoNLPy integration system for Korean morphological analysis.

    Provides unified interface to multiple Korean morphological analyzers
    with fallback mechanisms and performance optimization.
    """

    def __init__(self,
                 primary_analyzer: AnalyzerType = AnalyzerType.MECAB,
                 fallback_analyzers: Optional[List[AnalyzerType]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize KoNLPy integration.

        Args:
            primary_analyzer: Primary analyzer to use
            fallback_analyzers: List of fallback analyzers
            config: Configuration parameters
        """
        self.primary_analyzer = primary_analyzer
        self.fallback_analyzers = fallback_analyzers or [AnalyzerType.OKT, AnalyzerType.KOMORAN]
        self.config = config or {}

        self.analyzers = {}
        self.analyzer_status = {}
        self.performance_stats = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize all analyzers with fallback handling.

        Returns:
            bool: True if at least one analyzer is available
        """
        try:
            all_analyzers = [self.primary_analyzer] + self.fallback_analyzers
            initialized_count = 0

            for analyzer_type in all_analyzers:
                success = await self._initialize_analyzer(analyzer_type)
                if success:
                    initialized_count += 1

            self._initialized = initialized_count > 0

            if self._initialized:
                logger.info(f"KoNLPy integration initialized with {initialized_count} analyzers")
                return True
            else:
                logger.error("No KoNLPy analyzers could be initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize KoNLPy integration: {e}")
            return False

    async def _initialize_analyzer(self, analyzer_type: AnalyzerType) -> bool:
        """
        Initialize specific analyzer.

        Args:
            analyzer_type: Type of analyzer to initialize

        Returns:
            bool: True if initialization successful
        """
        try:
            analyzer = None

            if analyzer_type == AnalyzerType.MECAB:
                try:
                    from konlpy.tag import Mecab
                    analyzer = Mecab()
                except ImportError:
                    logger.warning("Mecab not available - requires system installation")
                    return False
                except Exception as e:
                    logger.warning(f"Mecab initialization failed: {e}")
                    return False

            elif analyzer_type == AnalyzerType.HANNANUM:
                try:
                    from konlpy.tag import Hannanum
                    analyzer = Hannanum()
                except ImportError:
                    logger.warning("Hannanum not available")
                    return False

            elif analyzer_type == AnalyzerType.KKMA:
                try:
                    from konlpy.tag import Kkma
                    analyzer = Kkma()
                except ImportError:
                    logger.warning("Kkma not available")
                    return False

            elif analyzer_type == AnalyzerType.KOMORAN:
                try:
                    from konlpy.tag import Komoran
                    analyzer = Komoran()
                except ImportError:
                    logger.warning("Komoran not available")
                    return False

            elif analyzer_type == AnalyzerType.OKT:
                try:
                    from konlpy.tag import Okt
                    analyzer = Okt()
                except ImportError:
                    logger.warning("Okt not available")
                    return False

            if analyzer:
                self.analyzers[analyzer_type] = analyzer
                self.analyzer_status[analyzer_type] = 'healthy'
                self.performance_stats[analyzer_type] = {
                    'total_calls': 0,
                    'total_time': 0.0,
                    'average_time': 0.0,
                    'error_count': 0
                }

                logger.info(f"Successfully initialized {analyzer_type.value} analyzer")
                return True

        except Exception as e:
            logger.error(f"Error initializing {analyzer_type.value}: {e}")
            self.analyzer_status[analyzer_type] = 'error'
            return False

        return False

    async def analyze_morphology(self,
                               text: str,
                               analyzer_type: Optional[AnalyzerType] = None,
                               include_features: bool = True) -> List[MorphemeResult]:
        """
        Perform morphological analysis on Korean text.

        Args:
            text: Korean text to analyze
            analyzer_type: Specific analyzer to use (None for auto-selection)
            include_features: Include additional morphological features

        Returns:
            List[MorphemeResult]: Morphological analysis results
        """
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            raise RuntimeError("No analyzers available for morphological analysis")

        # Select analyzer
        if analyzer_type and analyzer_type in self.analyzers:
            selected_analyzer = analyzer_type
        else:
            selected_analyzer = self._select_best_analyzer()

        # Perform analysis
        start_time = time.time()

        try:
            analyzer = self.analyzers[selected_analyzer]
            morphemes = []

            # Get morphemes with POS tags
            pos_tags = analyzer.pos(text)

            for surface, pos in pos_tags:
                features = {}

                if include_features:
                    # Add analyzer-specific features
                    features = await self._extract_features(surface, pos, selected_analyzer)

                morpheme = MorphemeResult(
                    surface=surface,
                    lemma=surface,  # Basic form - could be enhanced with lemmatization
                    pos=pos,
                    features=features
                )

                morphemes.append(morpheme)

            # Update performance stats
            processing_time = time.time() - start_time
            await self._update_performance_stats(selected_analyzer, processing_time, success=True)

            logger.debug(f"Morphology analysis completed using {selected_analyzer.value}: "
                        f"{len(morphemes)} morphemes in {processing_time:.3f}s")

            return morphemes

        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_performance_stats(selected_analyzer, processing_time, success=False)

            logger.error(f"Morphology analysis failed with {selected_analyzer.value}: {e}")

            # Try fallback analyzer
            fallback_result = await self._try_fallback_analysis(text, selected_analyzer, include_features)
            if fallback_result:
                return fallback_result

            raise RuntimeError(f"All morphological analyzers failed for text: {text[:50]}...")

    async def tokenize_text(self,
                          text: str,
                          analyzer_type: Optional[AnalyzerType] = None) -> List[str]:
        """
        Tokenize Korean text into morphemes.

        Args:
            text: Korean text to tokenize
            analyzer_type: Specific analyzer to use

        Returns:
            List[str]: List of tokens
        """
        if not self._initialized:
            await self.initialize()

        selected_analyzer = analyzer_type or self._select_best_analyzer()

        try:
            analyzer = self.analyzers[selected_analyzer]
            tokens = analyzer.morphs(text)

            logger.debug(f"Tokenization completed: {len(tokens)} tokens")
            return tokens

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return text.split()  # Fallback to simple space splitting

    async def extract_nouns(self,
                          text: str,
                          analyzer_type: Optional[AnalyzerType] = None) -> List[str]:
        """
        Extract nouns from Korean text.

        Args:
            text: Korean text to analyze
            analyzer_type: Specific analyzer to use

        Returns:
            List[str]: List of extracted nouns
        """
        if not self._initialized:
            await self.initialize()

        selected_analyzer = analyzer_type or self._select_best_analyzer()

        try:
            analyzer = self.analyzers[selected_analyzer]
            nouns = analyzer.nouns(text)

            logger.debug(f"Noun extraction completed: {len(nouns)} nouns")
            return nouns

        except Exception as e:
            logger.error(f"Noun extraction failed: {e}")
            return []

    def _select_best_analyzer(self) -> AnalyzerType:
        """
        Select the best available analyzer based on performance stats.

        Returns:
            AnalyzerType: Best analyzer to use
        """
        if not self.analyzers:
            raise RuntimeError("No analyzers available")

        # Prefer primary analyzer if available and healthy
        if (self.primary_analyzer in self.analyzers and
            self.analyzer_status.get(self.primary_analyzer) == 'healthy'):
            return self.primary_analyzer

        # Find best fallback analyzer
        best_analyzer = None
        best_score = float('inf')

        for analyzer_type in self.analyzers:
            if self.analyzer_status.get(analyzer_type) != 'healthy':
                continue

            stats = self.performance_stats.get(analyzer_type, {})

            # Score based on speed and error rate
            avg_time = stats.get('average_time', 1.0)
            error_rate = stats.get('error_count', 0) / max(stats.get('total_calls', 1), 1)

            score = avg_time * (1 + error_rate * 10)  # Penalize errors heavily

            if score < best_score:
                best_score = score
                best_analyzer = analyzer_type

        return best_analyzer or list(self.analyzers.keys())[0]

    async def _extract_features(self,
                              surface: str,
                              pos: str,
                              analyzer_type: AnalyzerType) -> Dict[str, Any]:
        """
        Extract additional features for morpheme.

        Args:
            surface: Surface form
            pos: Part-of-speech tag
            analyzer_type: Analyzer used

        Returns:
            Dict[str, Any]: Additional features
        """
        features = {
            'analyzer': analyzer_type.value,
            'length': len(surface),
            'is_korean': self._is_korean_text(surface),
            'pos_category': self._categorize_pos(pos)
        }

        return features

    def _is_korean_text(self, text: str) -> bool:
        """Check if text contains Korean characters."""
        korean_ranges = [
            (0xAC00, 0xD7AF),  # Hangul syllables
            (0x1100, 0x11FF),  # Hangul jamo
            (0x3130, 0x318F),  # Hangul compatibility jamo
        ]

        return any(
            any(start <= ord(char) <= end for start, end in korean_ranges)
            for char in text
        )

    def _categorize_pos(self, pos: str) -> str:
        """Categorize POS tag into major categories."""
        if pos.startswith('N'):
            return 'noun'
        elif pos.startswith('V'):
            return 'verb'
        elif pos.startswith('M'):
            return 'modifier'
        elif pos.startswith('J'):
            return 'particle'
        elif pos.startswith('E'):
            return 'ending'
        else:
            return 'other'

    async def _try_fallback_analysis(self,
                                   text: str,
                                   failed_analyzer: AnalyzerType,
                                   include_features: bool) -> Optional[List[MorphemeResult]]:
        """
        Try fallback analyzers when primary fails.

        Args:
            text: Text to analyze
            failed_analyzer: Analyzer that failed
            include_features: Include additional features

        Returns:
            Optional[List[MorphemeResult]]: Fallback analysis results
        """
        available_analyzers = [
            analyzer for analyzer in self.analyzers.keys()
            if analyzer != failed_analyzer and self.analyzer_status.get(analyzer) == 'healthy'
        ]

        for fallback_analyzer in available_analyzers:
            try:
                logger.info(f"Trying fallback analyzer: {fallback_analyzer.value}")
                return await self.analyze_morphology(text, fallback_analyzer, include_features)
            except Exception as e:
                logger.warning(f"Fallback analyzer {fallback_analyzer.value} also failed: {e}")

        return None

    async def _update_performance_stats(self,
                                      analyzer_type: AnalyzerType,
                                      processing_time: float,
                                      success: bool) -> None:
        """
        Update performance statistics for analyzer.

        Args:
            analyzer_type: Analyzer type
            processing_time: Time taken for processing
            success: Whether the operation was successful
        """
        stats = self.performance_stats.get(analyzer_type, {
            'total_calls': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'error_count': 0
        })

        stats['total_calls'] += 1
        stats['total_time'] += processing_time
        stats['average_time'] = stats['total_time'] / stats['total_calls']

        if not success:
            stats['error_count'] += 1

        self.performance_stats[analyzer_type] = stats

    async def get_analyzer_status(self) -> Dict[str, Any]:
        """
        Get status information for all analyzers.

        Returns:
            Dict[str, Any]: Status information
        """
        status = {
            'initialized': self._initialized,
            'primary_analyzer': self.primary_analyzer.value,
            'available_analyzers': [a.value for a in self.analyzers.keys()],
            'analyzer_status': {a.value: status for a, status in self.analyzer_status.items()},
            'performance_stats': {
                a.value: stats for a, stats in self.performance_stats.items()
            }
        }

        return status

    async def benchmark_analyzers(self, test_text: str = "안녕하세요. 한국어 형태소 분석을 테스트합니다.") -> Dict[str, Any]:
        """
        Benchmark all available analyzers.

        Args:
            test_text: Text to use for benchmarking

        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {}

        for analyzer_type in self.analyzers:
            if self.analyzer_status.get(analyzer_type) != 'healthy':
                continue

            try:
                start_time = time.time()
                morphemes = await self.analyze_morphology(test_text, analyzer_type)
                end_time = time.time()

                results[analyzer_type.value] = {
                    'processing_time': end_time - start_time,
                    'morpheme_count': len(morphemes),
                    'success': True
                }

            except Exception as e:
                results[analyzer_type.value] = {
                    'error': str(e),
                    'success': False
                }

        return results