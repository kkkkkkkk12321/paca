"""
Module: integrations.nlp.syntax_parser
Purpose: Korean syntax parsing with dependency analysis and phrase structure
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

from .morphology_analyzer import MorphologyAnalyzer, MorphologyResult

logger = logging.getLogger(__name__)

class SyntacticRole(Enum):
    """Syntactic roles in Korean sentences."""
    SUBJECT = "subject"        # 주어
    OBJECT = "object"         # 목적어
    COMPLEMENT = "complement" # 보어
    ADVERBIAL = "adverbial"   # 부사어
    MODIFIER = "modifier"     # 관형어
    PREDICATE = "predicate"   # 서술어
    VOCATIVE = "vocative"     # 호격어

class PhraseType(Enum):
    """Types of phrases in Korean."""
    NOUN_PHRASE = "noun_phrase"         # 명사구
    VERB_PHRASE = "verb_phrase"         # 동사구
    ADJECTIVE_PHRASE = "adjective_phrase" # 형용사구
    ADVERB_PHRASE = "adverb_phrase"     # 부사구
    PREPOSITIONAL_PHRASE = "prepositional_phrase" # 부사구 (전치사구에 해당)

class DependencyType(Enum):
    """Types of dependency relations."""
    HEAD = "head"             # 지배소
    DEPENDENT = "dependent"   # 의존소
    COORDINATION = "coordination" # 등위
    SUBORDINATION = "subordination" # 종속

@dataclass
class SyntacticNode:
    """Node in syntactic tree representing a word or phrase."""
    id: int
    text: str
    morphology: Optional[MorphologyResult]
    syntactic_role: Optional[SyntacticRole]
    phrase_type: Optional[PhraseType]
    head: Optional[int] = None  # ID of head node
    dependents: List[int] = None  # IDs of dependent nodes
    features: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependents is None:
            self.dependents = []
        if self.features is None:
            self.features = {}

@dataclass
class SyntacticTree:
    """Complete syntactic analysis tree."""
    nodes: List[SyntacticNode]
    root_id: Optional[int]
    sentence: str
    phrases: List[Dict[str, Any]] = None
    dependencies: List[Dict[str, Any]] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.phrases is None:
            self.phrases = []
        if self.dependencies is None:
            self.dependencies = []

class SyntaxParser:
    """
    Korean syntax parser with dependency analysis and phrase structure detection.

    Provides comprehensive syntactic analysis including dependency parsing,
    phrase structure analysis, and syntactic role assignment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize syntax parser.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.morphology_analyzer = MorphologyAnalyzer(config.get('morphology', {}))

        # Load syntactic patterns and rules
        self.particle_roles = self._load_particle_roles()
        self.phrase_patterns = self._load_phrase_patterns()
        self.dependency_rules = self._load_dependency_rules()

        self._initialized = False

    def _load_particle_roles(self) -> Dict[str, SyntacticRole]:
        """Load Korean particle to syntactic role mappings."""
        return {
            # Subject particles
            '이': SyntacticRole.SUBJECT,
            '가': SyntacticRole.SUBJECT,
            '께서': SyntacticRole.SUBJECT,

            # Object particles
            '을': SyntacticRole.OBJECT,
            '를': SyntacticRole.OBJECT,

            # Complement particles
            '이': SyntacticRole.COMPLEMENT,
            '가': SyntacticRole.COMPLEMENT,

            # Adverbial particles
            '에': SyntacticRole.ADVERBIAL,
            '에서': SyntacticRole.ADVERBIAL,
            '로': SyntacticRole.ADVERBIAL,
            '으로': SyntacticRole.ADVERBIAL,
            '와': SyntacticRole.ADVERBIAL,
            '과': SyntacticRole.ADVERBIAL,
            '하고': SyntacticRole.ADVERBIAL,

            # Modifier particles
            '의': SyntacticRole.MODIFIER,

            # Vocative particles
            '아': SyntacticRole.VOCATIVE,
            '야': SyntacticRole.VOCATIVE,
        }

    def _load_phrase_patterns(self) -> Dict[str, List[str]]:
        """Load phrase structure patterns."""
        return {
            'noun_phrase': [
                'MM + NNG',      # 관형사 + 명사
                'NNG + JKG + NNG', # 명사 + 의 + 명사
                'VV + ETN',      # 동사 + 명사형어미
                'VA + ETN'       # 형용사 + 명사형어미
            ],
            'verb_phrase': [
                'VV + EP + EF',  # 동사 + 선어말어미 + 종결어미
                'VV + EF',       # 동사 + 종결어미
                'VX + VV'        # 보조동사 + 동사
            ],
            'adjective_phrase': [
                'MAG + VA',      # 부사 + 형용사
                'VA + EP + EF'   # 형용사 + 선어말어미 + 종결어미
            ],
            'adverb_phrase': [
                'MAG + MAG',     # 부사 + 부사
                'NNG + JKB'      # 명사 + 부사격조사
            ]
        }

    def _load_dependency_rules(self) -> List[Dict[str, Any]]:
        """Load dependency parsing rules."""
        return [
            # Subject-predicate dependency
            {
                'pattern': ['SUBJECT', 'PREDICATE'],
                'head': 'PREDICATE',
                'dependent': 'SUBJECT',
                'relation': 'nsubj'
            },
            # Object-predicate dependency
            {
                'pattern': ['OBJECT', 'PREDICATE'],
                'head': 'PREDICATE',
                'dependent': 'OBJECT',
                'relation': 'obj'
            },
            # Modifier-noun dependency
            {
                'pattern': ['MODIFIER', 'NOUN'],
                'head': 'NOUN',
                'dependent': 'MODIFIER',
                'relation': 'nmod'
            },
            # Adverb-verb dependency
            {
                'pattern': ['ADVERBIAL', 'PREDICATE'],
                'head': 'PREDICATE',
                'dependent': 'ADVERBIAL',
                'relation': 'advmod'
            }
        ]

    async def initialize(self) -> bool:
        """
        Initialize the syntax parser.

        Returns:
            bool: True if initialization successful
        """
        try:
            await self.morphology_analyzer.initialize()
            self._initialized = True
            logger.info("Syntax parser initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize syntax parser: {e}")
            return False

    async def parse(self,
                   text: str,
                   include_phrases: bool = True,
                   include_dependencies: bool = True,
                   include_roles: bool = True) -> SyntacticTree:
        """
        Parse Korean text and generate syntactic tree.

        Args:
            text: Korean text to parse
            include_phrases: Include phrase structure analysis
            include_dependencies: Include dependency analysis
            include_roles: Include syntactic role assignment

        Returns:
            SyntacticTree: Complete syntactic analysis
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Get morphological analysis first
            morphology_results = await self.morphology_analyzer.analyze(text)

            # Create syntactic nodes
            nodes = await self._create_syntactic_nodes(morphology_results, text)

            # Assign syntactic roles
            if include_roles:
                await self._assign_syntactic_roles(nodes)

            # Find phrase structures
            phrases = []
            if include_phrases:
                phrases = await self._identify_phrases(nodes)

            # Build dependency structure
            dependencies = []
            root_id = None
            if include_dependencies:
                dependencies, root_id = await self._build_dependency_tree(nodes)

            # Create syntactic tree
            tree = SyntacticTree(
                nodes=nodes,
                root_id=root_id,
                sentence=text,
                phrases=phrases,
                dependencies=dependencies,
                confidence=self._calculate_parse_confidence(nodes, phrases, dependencies)
            )

            processing_time = time.time() - start_time
            logger.debug(f"Syntax parsing completed: {len(nodes)} nodes in {processing_time:.3f}s")

            return tree

        except Exception as e:
            logger.error(f"Syntax parsing failed: {e}")
            raise

    async def _create_syntactic_nodes(self,
                                    morphology_results: List[MorphologyResult],
                                    text: str) -> List[SyntacticNode]:
        """
        Create syntactic nodes from morphological analysis.

        Args:
            morphology_results: Morphological analysis results
            text: Original text

        Returns:
            List[SyntacticNode]: Syntactic nodes
        """
        nodes = []

        for i, morphology in enumerate(morphology_results):
            node = SyntacticNode(
                id=i,
                text=morphology.surface_form,
                morphology=morphology,
                syntactic_role=None,
                phrase_type=None,
                features={
                    'pos': morphology.pos_tag,
                    'word_class': morphology.word_class.value,
                    'lemma': morphology.lemma
                }
            )
            nodes.append(node)

        return nodes

    async def _assign_syntactic_roles(self, nodes: List[SyntacticNode]) -> None:
        """
        Assign syntactic roles to nodes based on particles and patterns.

        Args:
            nodes: Syntactic nodes to analyze
        """
        for i, node in enumerate(nodes):
            # Check for particle-based role assignment
            if node.morphology.word_class.value == 'particle':
                particle = node.text
                if particle in self.particle_roles:
                    # Assign role to previous noun
                    if i > 0 and nodes[i-1].morphology.word_class.value == 'noun':
                        nodes[i-1].syntactic_role = self.particle_roles[particle]
                        node.features['role_marker'] = True

            # Identify predicates
            elif node.morphology.word_class.value in ['verb', 'adjective']:
                # Check if it's a sentence-final predicate
                if self._is_sentence_final(node, nodes, i):
                    node.syntactic_role = SyntacticRole.PREDICATE

            # Identify modifiers
            elif node.morphology.word_class.value == 'modifier':
                node.syntactic_role = SyntacticRole.MODIFIER

    def _is_sentence_final(self, node: SyntacticNode, nodes: List[SyntacticNode], position: int) -> bool:
        """
        Check if a node is in sentence-final position.

        Args:
            node: Node to check
            nodes: All nodes
            position: Position of node

        Returns:
            bool: True if sentence-final
        """
        # Check if followed only by sentence-ending particles or punctuation
        for j in range(position + 1, len(nodes)):
            next_node = nodes[j]
            if (next_node.morphology.word_class.value not in ['particle', 'ending'] and
                not self._is_punctuation(next_node.text)):
                return False

        return True

    def _is_punctuation(self, text: str) -> bool:
        """Check if text is punctuation."""
        return text in '.?!,;:'

    async def _identify_phrases(self, nodes: List[SyntacticNode]) -> List[Dict[str, Any]]:
        """
        Identify phrase structures in the syntactic tree.

        Args:
            nodes: Syntactic nodes

        Returns:
            List[Dict[str, Any]]: Identified phrases
        """
        phrases = []
        used_nodes = set()

        # Identify noun phrases
        noun_phrases = await self._find_noun_phrases(nodes, used_nodes)
        phrases.extend(noun_phrases)

        # Identify verb phrases
        verb_phrases = await self._find_verb_phrases(nodes, used_nodes)
        phrases.extend(verb_phrases)

        # Identify adjective phrases
        adj_phrases = await self._find_adjective_phrases(nodes, used_nodes)
        phrases.extend(adj_phrases)

        return phrases

    async def _find_noun_phrases(self, nodes: List[SyntacticNode], used_nodes: Set[int]) -> List[Dict[str, Any]]:
        """Find noun phrases in the sentence."""
        noun_phrases = []

        for i, node in enumerate(nodes):
            if i in used_nodes or node.morphology.word_class.value != 'noun':
                continue

            phrase_nodes = [i]
            phrase_start = i
            phrase_end = i

            # Look backward for modifiers
            j = i - 1
            while j >= 0 and j not in used_nodes:
                prev_node = nodes[j]
                if (prev_node.morphology.word_class.value == 'modifier' or
                    (prev_node.morphology.word_class.value == 'noun' and
                     j + 1 < len(nodes) and nodes[j + 1].text == '의')):
                    phrase_nodes.insert(0, j)
                    phrase_start = j
                    j -= 1
                else:
                    break

            # Look forward for particles or additional nouns
            j = i + 1
            while j < len(nodes) and j not in used_nodes:
                next_node = nodes[j]
                if (next_node.morphology.word_class.value == 'particle' and
                    next_node.text in ['이', '가', '을', '를', '의']):
                    phrase_nodes.append(j)
                    phrase_end = j
                    j += 1
                else:
                    break

            if len(phrase_nodes) > 1:  # Only create phrase if it has multiple components
                phrase_text = ' '.join(nodes[k].text for k in phrase_nodes)
                noun_phrases.append({
                    'type': PhraseType.NOUN_PHRASE.value,
                    'text': phrase_text,
                    'start_node': phrase_start,
                    'end_node': phrase_end,
                    'node_ids': phrase_nodes,
                    'head_node': i  # The main noun is the head
                })

                # Mark nodes as used
                used_nodes.update(phrase_nodes)

        return noun_phrases

    async def _find_verb_phrases(self, nodes: List[SyntacticNode], used_nodes: Set[int]) -> List[Dict[str, Any]]:
        """Find verb phrases in the sentence."""
        verb_phrases = []

        for i, node in enumerate(nodes):
            if i in used_nodes or node.morphology.word_class.value != 'verb':
                continue

            phrase_nodes = [i]
            phrase_start = i
            phrase_end = i

            # Look forward for auxiliary verbs, endings, etc.
            j = i + 1
            while j < len(nodes) and j not in used_nodes:
                next_node = nodes[j]
                if (next_node.morphology.word_class.value in ['verb', 'ending'] or
                    next_node.morphology.pos_tag.startswith('E')):  # Endings
                    phrase_nodes.append(j)
                    phrase_end = j
                    j += 1
                else:
                    break

            # Look backward for adverbs modifying the verb
            j = i - 1
            while j >= 0 and j not in used_nodes:
                prev_node = nodes[j]
                if prev_node.morphology.word_class.value == 'adverb':
                    phrase_nodes.insert(0, j)
                    phrase_start = j
                    j -= 1
                else:
                    break

            if len(phrase_nodes) > 1:
                phrase_text = ' '.join(nodes[k].text for k in phrase_nodes)
                verb_phrases.append({
                    'type': PhraseType.VERB_PHRASE.value,
                    'text': phrase_text,
                    'start_node': phrase_start,
                    'end_node': phrase_end,
                    'node_ids': phrase_nodes,
                    'head_node': i  # The main verb is the head
                })

                used_nodes.update(phrase_nodes)

        return verb_phrases

    async def _find_adjective_phrases(self, nodes: List[SyntacticNode], used_nodes: Set[int]) -> List[Dict[str, Any]]:
        """Find adjective phrases in the sentence."""
        adj_phrases = []

        for i, node in enumerate(nodes):
            if i in used_nodes or node.morphology.word_class.value != 'adjective':
                continue

            phrase_nodes = [i]
            phrase_start = i
            phrase_end = i

            # Look backward for degree adverbs
            j = i - 1
            while j >= 0 and j not in used_nodes:
                prev_node = nodes[j]
                if (prev_node.morphology.word_class.value == 'adverb' and
                    prev_node.text in ['매우', '아주', '정말', '너무', '조금', '약간']):
                    phrase_nodes.insert(0, j)
                    phrase_start = j
                    j -= 1
                else:
                    break

            # Look forward for endings
            j = i + 1
            while j < len(nodes) and j not in used_nodes:
                next_node = nodes[j]
                if next_node.morphology.word_class.value == 'ending':
                    phrase_nodes.append(j)
                    phrase_end = j
                    j += 1
                else:
                    break

            if len(phrase_nodes) > 1:
                phrase_text = ' '.join(nodes[k].text for k in phrase_nodes)
                adj_phrases.append({
                    'type': PhraseType.ADJECTIVE_PHRASE.value,
                    'text': phrase_text,
                    'start_node': phrase_start,
                    'end_node': phrase_end,
                    'node_ids': phrase_nodes,
                    'head_node': i  # The adjective is the head
                })

                used_nodes.update(phrase_nodes)

        return adj_phrases

    async def _build_dependency_tree(self, nodes: List[SyntacticNode]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Build dependency tree structure.

        Args:
            nodes: Syntactic nodes

        Returns:
            Tuple[List[Dict[str, Any]], Optional[int]]: Dependencies and root ID
        """
        dependencies = []
        root_id = None

        # Find the main predicate as root
        for i, node in enumerate(nodes):
            if node.syntactic_role == SyntacticRole.PREDICATE:
                root_id = i
                break

        if root_id is None and nodes:
            # Fallback: use last verb/adjective as root
            for i in range(len(nodes) - 1, -1, -1):
                if nodes[i].morphology.word_class.value in ['verb', 'adjective']:
                    root_id = i
                    break

        # Build dependencies based on syntactic roles
        for i, node in enumerate(nodes):
            if node.syntactic_role and i != root_id:
                # Find appropriate head for this dependent
                head_id = await self._find_dependency_head(node, nodes, i, root_id)

                if head_id is not None:
                    # Create dependency relation
                    relation = self._determine_dependency_relation(node, nodes[head_id])

                    dependency = {
                        'dependent': i,
                        'head': head_id,
                        'relation': relation,
                        'dependent_text': node.text,
                        'head_text': nodes[head_id].text
                    }

                    dependencies.append(dependency)

                    # Update node structures
                    nodes[head_id].dependents.append(i)
                    nodes[i].head = head_id

        return dependencies, root_id

    async def _find_dependency_head(self,
                                  node: SyntacticNode,
                                  nodes: List[SyntacticNode],
                                  position: int,
                                  root_id: Optional[int]) -> Optional[int]:
        """
        Find the head for a dependent node.

        Args:
            node: Dependent node
            nodes: All nodes
            position: Position of dependent node
            root_id: Root node ID

        Returns:
            Optional[int]: Head node ID
        """
        if not node.syntactic_role:
            return None

        # Subject and object depend on predicate
        if node.syntactic_role in [SyntacticRole.SUBJECT, SyntacticRole.OBJECT, SyntacticRole.ADVERBIAL]:
            if root_id is not None:
                return root_id

            # Find nearest predicate
            for i in range(len(nodes)):
                if nodes[i].syntactic_role == SyntacticRole.PREDICATE:
                    return i

        # Modifier depends on following noun
        elif node.syntactic_role == SyntacticRole.MODIFIER:
            for i in range(position + 1, len(nodes)):
                if nodes[i].morphology.word_class.value == 'noun':
                    return i

        return None

    def _determine_dependency_relation(self, dependent: SyntacticNode, head: SyntacticNode) -> str:
        """
        Determine the dependency relation label.

        Args:
            dependent: Dependent node
            head: Head node

        Returns:
            str: Dependency relation label
        """
        if dependent.syntactic_role == SyntacticRole.SUBJECT:
            return 'nsubj'  # nominal subject
        elif dependent.syntactic_role == SyntacticRole.OBJECT:
            return 'obj'    # direct object
        elif dependent.syntactic_role == SyntacticRole.MODIFIER:
            return 'nmod'   # nominal modifier
        elif dependent.syntactic_role == SyntacticRole.ADVERBIAL:
            return 'advmod' # adverbial modifier
        else:
            return 'dep'    # generic dependency

    def _calculate_parse_confidence(self,
                                  nodes: List[SyntacticNode],
                                  phrases: List[Dict[str, Any]],
                                  dependencies: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the parse.

        Args:
            nodes: Syntactic nodes
            phrases: Identified phrases
            dependencies: Dependency relations

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not nodes:
            return 0.0

        # Base confidence from morphological analysis
        morphology_confidence = sum(node.morphology.confidence for node in nodes) / len(nodes)

        # Bonus for successful role assignment
        role_assigned = sum(1 for node in nodes if node.syntactic_role)
        role_ratio = role_assigned / len(nodes)

        # Bonus for phrase identification
        phrase_coverage = len(phrases) / max(len(nodes) // 3, 1)  # Rough heuristic

        # Bonus for dependency structure
        dependency_ratio = len(dependencies) / max(len(nodes) - 1, 1)

        # Combine scores
        confidence = (morphology_confidence * 0.4 +
                     role_ratio * 0.3 +
                     min(phrase_coverage, 1.0) * 0.2 +
                     min(dependency_ratio, 1.0) * 0.1)

        return min(confidence, 1.0)

    async def get_parse_statistics(self, tree: SyntacticTree) -> Dict[str, Any]:
        """
        Get statistics for syntactic parse.

        Args:
            tree: Syntactic tree

        Returns:
            Dict[str, Any]: Parse statistics
        """
        stats = {
            'total_nodes': len(tree.nodes),
            'syntactic_roles': {},
            'phrase_types': {},
            'dependency_relations': {},
            'parse_confidence': tree.confidence,
            'has_root': tree.root_id is not None
        }

        # Count syntactic roles
        for node in tree.nodes:
            if node.syntactic_role:
                role = node.syntactic_role.value
                stats['syntactic_roles'][role] = stats['syntactic_roles'].get(role, 0) + 1

        # Count phrase types
        for phrase in tree.phrases:
            phrase_type = phrase['type']
            stats['phrase_types'][phrase_type] = stats['phrase_types'].get(phrase_type, 0) + 1

        # Count dependency relations
        for dep in tree.dependencies:
            relation = dep['relation']
            stats['dependency_relations'][relation] = stats['dependency_relations'].get(relation, 0) + 1

        return stats