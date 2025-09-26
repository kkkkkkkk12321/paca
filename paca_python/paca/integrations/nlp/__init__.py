"""
Module: integrations.nlp
Purpose: Korean Natural Language Processing integration system
Author: PACA Development Team
Created: 2024-09-24
Last Modified: 2024-09-24
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging

# Import all NLP components
from .konlpy_integration import KoNLPyIntegration
from .korean_tokenizer import KoreanTokenizer
from .morphology_analyzer import MorphologyAnalyzer
from .syntax_parser import SyntaxParser
from .semantic_analyzer import SemanticAnalyzer
from .cultural_context import CulturalContextProcessor

logger = logging.getLogger(__name__)

class KoreanNLPSystem:
    """
    Korean NLP System integrator for unified processing.

    This class provides a unified interface to all Korean NLP capabilities
    including tokenization, morphology analysis, syntax parsing, semantic analysis,
    and cultural context processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Korean NLP System.

        Args:
            config (Dict[str, Any], optional): Configuration parameters
        """
        self.config = config or {}
        self.tokenizer = None
        self.morphology_analyzer = None
        self.syntax_parser = None
        self.semantic_analyzer = None
        self.cultural_processor = None
        self._is_initialized = False

        logger.info("Korean NLP System initialized")

    async def initialize(self) -> bool:
        """
        Initialize all NLP components.

        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize tokenizer
            self.tokenizer = KoreanTokenizer(
                model_type=self.config.get('tokenizer_model', 'mecab')
            )

            # Initialize morphology analyzer
            self.morphology_analyzer = MorphologyAnalyzer(
                config=self.config.get('morphology', {})
            )

            # Initialize syntax parser
            self.syntax_parser = SyntaxParser(
                config=self.config.get('syntax', {})
            )

            # Initialize semantic analyzer
            self.semantic_analyzer = SemanticAnalyzer(
                config=self.config.get('semantic', {})
            )

            # Initialize cultural context processor
            self.cultural_processor = CulturalContextProcessor(
                config=self.config.get('cultural', {})
            )

            self._is_initialized = True
            logger.info("All Korean NLP components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Korean NLP system: {e}")
            return False

    async def process_text(self, text: str,
                          include_tokenization: bool = True,
                          include_morphology: bool = True,
                          include_syntax: bool = True,
                          include_semantics: bool = True,
                          include_cultural: bool = True) -> Dict[str, Any]:
        """
        Process Korean text through full NLP pipeline.

        Args:
            text (str): Input Korean text
            include_tokenization (bool): Include tokenization results
            include_morphology (bool): Include morphology analysis
            include_syntax (bool): Include syntax parsing
            include_semantics (bool): Include semantic analysis
            include_cultural (bool): Include cultural context

        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        if not self._is_initialized:
            await self.initialize()

        results = {
            'input_text': text,
            'timestamp': asyncio.get_event_loop().time(),
            'analysis': {}
        }

        try:
            # Tokenization
            if include_tokenization and self.tokenizer:
                tokens = await self.tokenizer.tokenize(text)
                results['analysis']['tokenization'] = tokens
                logger.debug(f"Tokenization completed: {len(tokens)} tokens")

            # Morphology analysis
            if include_morphology and self.morphology_analyzer:
                morphology = await self.morphology_analyzer.analyze(text)
                results['analysis']['morphology'] = morphology
                logger.debug("Morphology analysis completed")

            # Syntax parsing
            if include_syntax and self.syntax_parser:
                syntax = await self.syntax_parser.parse(text)
                results['analysis']['syntax'] = syntax
                logger.debug("Syntax parsing completed")

            # Semantic analysis
            if include_semantics and self.semantic_analyzer:
                semantics = await self.semantic_analyzer.analyze(text)
                results['analysis']['semantics'] = semantics
                logger.debug("Semantic analysis completed")

            # Cultural context
            if include_cultural and self.cultural_processor:
                cultural = await self.cultural_processor.analyze_context(text)
                results['analysis']['cultural_context'] = cultural
                logger.debug("Cultural context analysis completed")

            results['status'] = 'success'
            results['processing_time'] = asyncio.get_event_loop().time() - results['timestamp']

            return results

        except Exception as e:
            logger.error(f"Error in NLP processing pipeline: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all NLP components.

        Returns:
            Dict[str, Any]: Health status information
        """
        status = {
            'overall': 'healthy' if self._is_initialized else 'not_initialized',
            'components': {
                'tokenizer': 'healthy' if self.tokenizer else 'not_initialized',
                'morphology': 'healthy' if self.morphology_analyzer else 'not_initialized',
                'syntax': 'healthy' if self.syntax_parser else 'not_initialized',
                'semantic': 'healthy' if self.semantic_analyzer else 'not_initialized',
                'cultural': 'healthy' if self.cultural_processor else 'not_initialized'
            },
            'timestamp': asyncio.get_event_loop().time()
        }

        return status

# Export main components
__all__ = [
    'KoreanNLPSystem',
    'KoNLPyIntegration',
    'KoreanTokenizer',
    'MorphologyAnalyzer',
    'SyntaxParser',
    'SemanticAnalyzer',
    'CulturalContextProcessor'
]

# Global NLP system instance
_nlp_system = None

async def get_nlp_system(config: Optional[Dict[str, Any]] = None) -> KoreanNLPSystem:
    """
    Get global NLP system instance (singleton pattern).

    Args:
        config (Dict[str, Any], optional): Configuration parameters

    Returns:
        KoreanNLPSystem: Global NLP system instance
    """
    global _nlp_system

    if _nlp_system is None:
        _nlp_system = KoreanNLPSystem(config)
        await _nlp_system.initialize()

    return _nlp_system

async def quick_analyze(text: str, analysis_type: str = "full") -> Dict[str, Any]:
    """
    Quick analysis function for simple Korean NLP tasks.

    Args:
        text (str): Input Korean text
        analysis_type (str): Type of analysis ('tokenize', 'morphology', 'full')

    Returns:
        Dict[str, Any]: Analysis results
    """
    nlp = await get_nlp_system()

    if analysis_type == "tokenize":
        return await nlp.process_text(text,
                                    include_morphology=False,
                                    include_syntax=False,
                                    include_semantics=False,
                                    include_cultural=False)
    elif analysis_type == "morphology":
        return await nlp.process_text(text,
                                    include_syntax=False,
                                    include_semantics=False,
                                    include_cultural=False)
    else:  # full analysis
        return await nlp.process_text(text)