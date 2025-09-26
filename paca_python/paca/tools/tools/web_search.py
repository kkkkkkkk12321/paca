"""
웹 검색 도구 (The Scout)

최신 정보 수집 및 사실 확인을 위한 웹 검색 도구
"""

import aiohttp
import asyncio
import urllib.parse
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import json
import re

from ..base import Tool, ToolResult, ToolType


@dataclass
class SearchResult:
    """검색 결과 항목"""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    timestamp: Optional[datetime] = None
    source_type: str = "web"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'score': self.score,
            'timestamp': self.timestamp.isoformat(),
            'source_type': self.source_type,
            'metadata': self.metadata
        }


class WebSearchTool(Tool):
    """웹 검색 도구 - The Scout"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            tool_type=ToolType.SEARCH,
            description="웹에서 정보를 검색하고 최신 정보를 수집합니다."
        )
        self.api_key = api_key
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'bing': self._search_bing,
            'google': self._search_google
        }
        self.default_engine = 'duckduckgo'

    def validate_input(self, query: str = "", **kwargs) -> bool:
        """입력 검증"""
        if not query or not isinstance(query, str):
            return False
        if len(query.strip()) < 2:
            return False
        return True

    async def execute(self, query: str, max_results: int = 10,
                     search_engine: str = None, include_images: bool = False,
                     time_filter: str = None, **kwargs) -> ToolResult:
        """웹 검색 실행"""
        try:
            # 파라미터 검증
            if not self.validate_input(query):
                return ToolResult(
                    success=False,
                    error="유효하지 않은 검색 쿼리입니다."
                )

            query = query.strip()
            max_results = min(max_results, 50)  # 최대 50개로 제한
            search_engine = search_engine or self.default_engine

            # 검색 엔진 선택
            if search_engine not in self.search_engines:
                search_engine = self.default_engine

            # 검색 실행
            search_func = self.search_engines[search_engine]
            results = await search_func(
                query, max_results, include_images, time_filter
            )

            # 결과 후처리
            processed_results = self._process_results(results, query)

            metadata = {
                'query': query,
                'search_engine': search_engine,
                'result_count': len(processed_results),
                'max_results': max_results,
                'include_images': include_images,
                'time_filter': time_filter
            }

            return ToolResult(
                success=True,
                data=processed_results,
                metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"웹 검색 중 오류 발생: {str(e)}"
            )

    async def _search_duckduckgo(self, query: str, max_results: int,
                                include_images: bool, time_filter: str) -> List[SearchResult]:
        """DuckDuckGo 검색"""
        results = []

        try:
            # DuckDuckGo Instant Answer API 사용
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Abstract 결과
                        if data.get('Abstract'):
                            results.append(SearchResult(
                                title=data.get('Heading', 'DuckDuckGo Abstract'),
                                url=data.get('AbstractURL', ''),
                                snippet=data['Abstract'],
                                score=0.9,
                                source_type='abstract'
                            ))

                        # Related Topics
                        for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                            if isinstance(topic, dict) and 'Text' in topic:
                                results.append(SearchResult(
                                    title=topic.get('Text', '')[:100],
                                    url=topic.get('FirstURL', ''),
                                    snippet=topic.get('Text', ''),
                                    score=0.7,
                                    source_type='related'
                                ))

                        # Answer
                        if data.get('Answer'):
                            results.append(SearchResult(
                                title='Direct Answer',
                                url='',
                                snippet=data['Answer'],
                                score=1.0,
                                source_type='answer'
                            ))

            # 추가 검색이 필요한 경우 HTML 검색 시뮬레이션
            if len(results) < max_results:
                additional_results = await self._search_duckduckgo_html(
                    query, max_results - len(results)
                )
                results.extend(additional_results)

        except Exception as e:
            # 기본 검색 결과 제공
            results.append(SearchResult(
                title=f"검색어: {query}",
                url="",
                snippet=f"'{query}'에 대한 검색이 요청되었습니다. 외부 검색 API에 연결할 수 없어 시뮬레이션 결과를 제공합니다.",
                score=0.5,
                source_type='simulation'
            ))

        return results[:max_results]

    async def _search_duckduckgo_html(self, query: str, max_results: int) -> List[SearchResult]:
        """DuckDuckGo HTML 검색 (시뮬레이션)"""
        # 실제 구현에서는 requests-html 또는 selenium 사용
        # 여기서는 시뮬레이션된 결과 제공
        simulation_results = [
            SearchResult(
                title=f"{query} 관련 정보 #{i+1}",
                url=f"https://example.com/result_{i+1}",
                snippet=f"'{query}'와 관련된 유용한 정보를 포함하고 있는 웹 페이지입니다. 상세한 내용을 확인하시기 바랍니다.",
                score=0.8 - (i * 0.1),
                source_type='web'
            ) for i in range(min(max_results, 3))
        ]

        return simulation_results

    async def _search_bing(self, query: str, max_results: int,
                          include_images: bool, time_filter: str) -> List[SearchResult]:
        """Bing 검색 (API 키 필요)"""
        if not self.api_key:
            return [SearchResult(
                title="Bing 검색 API 키 필요",
                url="",
                snippet="Bing 검색을 사용하려면 API 키가 필요합니다.",
                score=0.3,
                source_type='error'
            )]

        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {'Ocp-Apim-Subscription-Key': self.api_key}
            params = {
                'q': query,
                'count': max_results,
                'mkt': 'ko-KR'
            }

            if time_filter:
                params['freshness'] = time_filter

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        for item in data.get('webPages', {}).get('value', []):
                            results.append(SearchResult(
                                title=item.get('name', ''),
                                url=item.get('url', ''),
                                snippet=item.get('snippet', ''),
                                score=0.8,
                                source_type='bing'
                            ))

                        return results
                    else:
                        return [SearchResult(
                            title="Bing API 오류",
                            url="",
                            snippet=f"Bing API 호출 실패: {response.status}",
                            score=0.2,
                            source_type='error'
                        )]

        except Exception as e:
            return [SearchResult(
                title="Bing 검색 오류",
                url="",
                snippet=f"Bing 검색 중 오류: {str(e)}",
                score=0.2,
                source_type='error'
            )]

    async def _search_google(self, query: str, max_results: int,
                           include_images: bool, time_filter: str) -> List[SearchResult]:
        """Google 검색 (Custom Search API 필요)"""
        if not self.api_key:
            return [SearchResult(
                title="Google 검색 API 키 필요",
                url="",
                snippet="Google 검색을 사용하려면 Custom Search API 키가 필요합니다.",
                score=0.3,
                source_type='error'
            )]

        # Google Custom Search API 구현 (시뮬레이션)
        return [SearchResult(
            title=f"Google 검색: {query}",
            url="https://www.google.com/search?q=" + urllib.parse.quote(query),
            snippet="Google Custom Search API 키를 설정하면 실제 검색 결과를 제공할 수 있습니다.",
            score=0.6,
            source_type='simulation'
        )]

    def _process_results(self, results: List[SearchResult], query: str) -> List[Dict[str, Any]]:
        """검색 결과 후처리"""
        processed = []

        for result in results:
            # 관련성 점수 계산
            relevance_score = self._calculate_relevance(result, query)
            result.score = relevance_score

            # 중복 제거
            if not self._is_duplicate(result, processed):
                processed.append(result.to_dict())

        # 점수순 정렬
        processed.sort(key=lambda x: x['score'], reverse=True)

        return processed

    def _calculate_relevance(self, result: SearchResult, query: str) -> float:
        """관련성 점수 계산"""
        score = result.score
        query_lower = query.lower()

        # 제목에서 검색어 포함 여부
        if query_lower in result.title.lower():
            score += 0.3

        # 스니펫에서 검색어 포함 여부
        if query_lower in result.snippet.lower():
            score += 0.2

        # 소스 타입별 가중치
        type_weights = {
            'answer': 1.0,
            'abstract': 0.9,
            'web': 0.8,
            'related': 0.7,
            'simulation': 0.5,
            'error': 0.1
        }

        score *= type_weights.get(result.source_type, 0.5)

        return min(score, 1.0)

    def _is_duplicate(self, result: SearchResult, processed: List[Dict[str, Any]]) -> bool:
        """중복 결과 확인"""
        for existing in processed:
            # URL이 같거나 제목이 매우 유사한 경우
            if (result.url and result.url == existing.get('url')) or \
               (self._similarity(result.title, existing.get('title', '')) > 0.8):
                return True
        return False

    def _similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (단순 버전)"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    async def search_and_verify(self, query: str, verification_sources: int = 3) -> ToolResult:
        """검색 및 교차 검증"""
        try:
            # 기본 검색
            result = await self.execute(query, max_results=verification_sources * 2)
            if not result.success:
                return result

            search_results = result.data

            # 상위 결과들로 교차 검증
            verified_info = {
                'query': query,
                'consensus': None,
                'confidence': 0.0,
                'sources': [],
                'conflicting_info': []
            }

            top_results = search_results[:verification_sources]

            # 간단한 일치성 검사
            common_keywords = self._extract_common_keywords(top_results)

            verified_info['consensus'] = f"'{query}'에 대한 검색 결과에서 공통적으로 언급되는 키워드: {', '.join(common_keywords[:5])}"
            verified_info['confidence'] = min(len(common_keywords) / 10.0, 1.0)
            verified_info['sources'] = [r['url'] for r in top_results if r['url']]

            return ToolResult(
                success=True,
                data=verified_info,
                metadata={'original_results': search_results}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"검색 및 검증 중 오류: {str(e)}"
            )

    def _extract_common_keywords(self, results: List[Dict[str, Any]]) -> List[str]:
        """공통 키워드 추출"""
        all_text = " ".join([
            f"{r.get('title', '')} {r.get('snippet', '')}"
            for r in results
        ])

        # 간단한 키워드 추출 (실제로는 NLP 라이브러리 사용 권장)
        words = re.findall(r'\b\w{3,}\b', all_text.lower())
        word_count = {}

        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # 빈도순 정렬
        common_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 불용어 제거 (간단한 버전)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'what', 'how', 'when', 'where', 'why', 'who'}

        filtered_words = [word for word, count in common_words
                         if word not in stopwords and count > 1]

        return filtered_words

    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        return {
            'tool_name': self.name,
            'supported_engines': list(self.search_engines.keys()),
            'default_engine': self.default_engine,
            'api_key_configured': bool(self.api_key),
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }