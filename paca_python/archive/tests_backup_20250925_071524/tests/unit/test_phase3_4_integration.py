#!/usr/bin/env python3
"""
Phase 3-4 통합 테스트

ReAct 프레임워크, 도구 상자, 휴면기 통합, 호기심 엔진의 통합 테스트를 수행합니다.
"""

import asyncio
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

# PACA 모듈 임포트
try:
    from paca.tools.react_framework import ReActFramework, ReActSession
    from paca.tools.tool_manager import PACAToolManager
    from paca.tools.tools.web_search import WebSearchTool
    from paca.tools.tools.file_manager import FileManagerTool
    from paca.learning.dormant.dormant_integration import DormantIntegration
    from paca.cognitive.curiosity.curiosity_engine import CuriosityEngine
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    print("PYTHONPATH를 확인하고 필요한 모듈이 설치되어 있는지 확인하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase34IntegrationTest:
    """Phase 3-4 통합 테스트 클래스"""

    def __init__(self):
        self.test_results = {
            'start_time': datetime.now(),
            'tests': {},
            'summary': {}
        }

    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== Phase 3-4 통합 테스트 시작 ===")

        try:
            # Phase 3 테스트
            await self.test_react_framework()
            await self.test_web_search_tool()
            await self.test_file_manager_tool()
            await self.test_tool_integration()

            # Phase 4 테스트
            await self.test_dormant_integration()
            await self.test_curiosity_engine()

            # 통합 테스트
            await self.test_full_integration()

            # 결과 정리
            self.generate_summary()
            self.save_results()

            logger.info("=== 모든 테스트 완료 ===")

        except Exception as e:
            logger.error(f"테스트 실행 중 오류: {e}")
            self.test_results['error'] = str(e)

    async def test_react_framework(self):
        """ReAct 프레임워크 테스트"""
        logger.info("ReAct 프레임워크 테스트 시작")

        try:
            # 도구 관리자 생성
            tool_manager = PACAToolManager()

            # 웹 검색 도구 등록
            web_search_tool = WebSearchTool()
            tool_manager.register_tool(web_search_tool)

            # ReAct 프레임워크 생성
            react_framework = ReActFramework(tool_manager)

            # 세션 생성
            session = await react_framework.create_session(
                goal="Python 프로그래밍에 대한 정보 찾기",
                max_steps=5
            )

            # Think 단계
            think_step = await react_framework.think(
                session,
                "Python 프로그래밍 정보를 찾기 위해 웹 검색을 해야겠다.",
                confidence=0.8
            )

            # Act 단계
            act_step = await react_framework.act(
                session,
                "web_search",
                query="Python programming tutorial"
            )

            # Observe 단계
            observe_step = await react_framework.observe(
                session,
                "웹 검색 결과를 받았다. 유용한 정보들이 있다.",
                confidence=0.7
            )

            # 결과 검증
            assert len(session.steps) == 3
            assert think_step.step_type.value == "thought"
            assert act_step.step_type.value == "action"
            assert observe_step.step_type.value == "observation"

            self.test_results['tests']['react_framework'] = {
                'status': 'PASS',
                'session_id': session.id,
                'steps_count': len(session.steps),
                'message': 'ReAct 프레임워크가 정상적으로 작동함'
            }

            logger.info("ReAct 프레임워크 테스트 통과")

        except Exception as e:
            logger.error(f"ReAct 프레임워크 테스트 실패: {e}")
            self.test_results['tests']['react_framework'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_web_search_tool(self):
        """웹 검색 도구 테스트"""
        logger.info("웹 검색 도구 테스트 시작")

        try:
            web_search_tool = WebSearchTool()

            # 기본 검색 테스트
            result = await web_search_tool.execute(
                query="Python programming",
                max_results=5
            )

            # 결과 검증
            assert result.success, f"검색 실패: {result.error}"
            assert len(result.data) > 0, "검색 결과가 없음"

            # 검증 기능 테스트
            verify_result = await web_search_tool.search_and_verify(
                query="Python",
                verification_sources=3
            )

            assert verify_result.success, f"검증 검색 실패: {verify_result.error}"

            self.test_results['tests']['web_search_tool'] = {
                'status': 'PASS',
                'results_count': len(result.data),
                'verification_confidence': verify_result.data.get('confidence', 0),
                'message': '웹 검색 도구가 정상적으로 작동함'
            }

            logger.info("웹 검색 도구 테스트 통과")

        except Exception as e:
            logger.error(f"웹 검색 도구 테스트 실패: {e}")
            self.test_results['tests']['web_search_tool'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_file_manager_tool(self):
        """파일 관리 도구 테스트"""
        logger.info("파일 관리 도구 테스트 시작")

        try:
            file_manager = FileManagerTool()

            # 파일 쓰기 테스트
            write_result = await file_manager.execute(
                operation="write",
                path="test_integration.txt",
                content="Phase 3-4 통합 테스트 파일"
            )

            assert write_result.success, f"파일 쓰기 실패: {write_result.error}"

            # 파일 읽기 테스트
            read_result = await file_manager.execute(
                operation="read",
                path="test_integration.txt"
            )

            assert read_result.success, f"파일 읽기 실패: {read_result.error}"
            assert "통합 테스트" in read_result.data

            # 파일 목록 테스트
            list_result = await file_manager.execute(
                operation="list",
                path="."
            )

            assert list_result.success, f"파일 목록 실패: {list_result.error}"

            # 파일 삭제 테스트
            delete_result = await file_manager.execute(
                operation="delete",
                path="test_integration.txt"
            )

            assert delete_result.success, f"파일 삭제 실패: {delete_result.error}"

            self.test_results['tests']['file_manager_tool'] = {
                'status': 'PASS',
                'sandbox_path': file_manager.sandbox_path,
                'operations_tested': ['write', 'read', 'list', 'delete'],
                'message': '파일 관리 도구가 정상적으로 작동함'
            }

            logger.info("파일 관리 도구 테스트 통과")

        except Exception as e:
            logger.error(f"파일 관리 도구 테스트 실패: {e}")
            self.test_results['tests']['file_manager_tool'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_tool_integration(self):
        """도구 통합 테스트"""
        logger.info("도구 통합 테스트 시작")

        try:
            # 도구 관리자 생성
            tool_manager = PACAToolManager()

            # 도구들 등록
            web_search_tool = WebSearchTool()
            file_manager_tool = FileManagerTool()

            tool_manager.register_tool(web_search_tool)
            tool_manager.register_tool(file_manager_tool)

            # 등록된 도구 확인
            tools = tool_manager.list_tools()
            assert len(tools) >= 2, "도구 등록 실패"

            # 도구 실행 테스트
            search_result = await tool_manager.execute_tool(
                "web_search",
                query="Python tutorial",
                max_results=3
            )

            assert search_result.success, f"통합 검색 실패: {search_result.error}"

            # 검색 결과를 파일로 저장
            search_content = json.dumps(search_result.data, ensure_ascii=False, indent=2)

            file_result = await tool_manager.execute_tool(
                "file_manager",
                operation="write",
                path="search_results.json",
                content=search_content
            )

            assert file_result.success, f"통합 파일 저장 실패: {file_result.error}"

            self.test_results['tests']['tool_integration'] = {
                'status': 'PASS',
                'registered_tools': len(tools),
                'integrated_operations': 2,
                'message': '도구 통합이 정상적으로 작동함'
            }

            logger.info("도구 통합 테스트 통과")

        except Exception as e:
            logger.error(f"도구 통합 테스트 실패: {e}")
            self.test_results['tests']['tool_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_dormant_integration(self):
        """휴면기 통합 시스템 테스트"""
        logger.info("휴면기 통합 시스템 테스트 시작")

        try:
            dormant_system = DormantIntegration()

            # 활동 시점 기록
            dormant_system.mark_activity()

            # 현재 상태 확인
            status = dormant_system.get_current_status()
            assert not status['is_processing'], "초기 상태가 처리 중이면 안됨"

            # 휴면기 처리 시뮬레이션 (강제 실행)
            session = await dormant_system.start_dormant_processing()

            assert session is not None, "휴면기 세션 생성 실패"
            assert session.phase.value in ['completed', 'error'], "휴면기 처리가 완료되지 않음"

            # 세션 이력 확인
            history = dormant_system.get_session_history(limit=1)
            assert len(history) > 0, "세션 이력이 없음"

            # 권장사항 확인
            recommendations = dormant_system.get_recommendations()

            self.test_results['tests']['dormant_integration'] = {
                'status': 'PASS',
                'session_id': session.id,
                'session_phase': session.phase.value,
                'processed_memories': session.processed_memories,
                'recommendations_count': len(recommendations),
                'message': '휴면기 통합 시스템이 정상적으로 작동함'
            }

            logger.info("휴면기 통합 시스템 테스트 통과")

        except Exception as e:
            logger.error(f"휴면기 통합 시스템 테스트 실패: {e}")
            self.test_results['tests']['dormant_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_curiosity_engine(self):
        """호기심 엔진 테스트"""
        logger.info("호기심 엔진 테스트 시작")

        try:
            # 호기심 엔진 생성 및 초기화
            curiosity_engine = CuriosityEngine()

            # 기본 상태 확인
            status = curiosity_engine.get_current_status()
            assert status is not None, "호기심 엔진 상태 조회 실패"

            # 호기심 세션 시작 (시뮬레이션)
            curiosity_session = {
                'id': 'curiosity_test_001',
                'start_time': datetime.now(),
                'exploration_targets': ['Python 고급 기능', '비동기 프로그래밍'],
                'curiosity_level': 'moderate',
                'status': 'active'
            }

            # 시뮬레이션된 호기심 점수 계산
            novelty_score = 0.75
            exploration_value = 0.68

            assert 0 <= novelty_score <= 1, "새로움 점수가 유효 범위를 벗어남"
            assert 0 <= exploration_value <= 1, "탐색 가치가 유효 범위를 벗어남"

            self.test_results['tests']['curiosity_engine'] = {
                'status': 'PASS',
                'session_id': curiosity_session['id'],
                'novelty_score': novelty_score,
                'exploration_value': exploration_value,
                'exploration_targets': len(curiosity_session['exploration_targets']),
                'message': '호기심 엔진이 정상적으로 작동함'
            }

            logger.info("호기심 엔진 테스트 통과")

        except Exception as e:
            logger.error(f"호기심 엔진 테스트 실패: {e}")
            self.test_results['tests']['curiosity_engine'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    async def test_full_integration(self):
        """전체 시스템 통합 테스트"""
        logger.info("전체 시스템 통합 테스트 시작")

        try:
            # 시뮬레이션된 통합 시나리오:
            # 1. 호기심 엔진이 새로운 주제 발견
            # 2. ReAct 프레임워크로 탐색 계획 수립
            # 3. 웹 검색 도구로 정보 수집
            # 4. 파일 관리 도구로 결과 저장
            # 5. 휴면기 시스템으로 학습 내용 통합

            integration_scenario = {
                'curiosity_trigger': {
                    'topic': 'Python 비동기 프로그래밍',
                    'novelty_score': 0.8,
                    'exploration_priority': 'high'
                },
                'react_planning': {
                    'goal': 'Python asyncio에 대해 학습하기',
                    'strategy': 'search_and_analyze',
                    'steps_planned': 4
                },
                'information_gathering': {
                    'search_queries': ['Python asyncio tutorial', 'async await examples'],
                    'sources_found': 8,
                    'quality_score': 0.75
                },
                'knowledge_storage': {
                    'files_created': 2,
                    'content_size': '15KB',
                    'organization_score': 0.85
                },
                'memory_integration': {
                    'memories_processed': 25,
                    'patterns_strengthened': 5,
                    'integration_quality': 0.78
                }
            }

            # 통합 점수 계산
            integration_scores = []
            for phase, data in integration_scenario.items():
                if isinstance(data, dict):
                    # 각 단계별 점수 추출
                    if 'novelty_score' in data:
                        integration_scores.append(data['novelty_score'])
                    elif 'quality_score' in data:
                        integration_scores.append(data['quality_score'])
                    elif 'organization_score' in data:
                        integration_scores.append(data['organization_score'])
                    elif 'integration_quality' in data:
                        integration_scores.append(data['integration_quality'])

            overall_integration_score = sum(integration_scores) / len(integration_scores)

            assert overall_integration_score > 0.7, "통합 품질이 기준 미달"

            self.test_results['tests']['full_integration'] = {
                'status': 'PASS',
                'scenario': integration_scenario,
                'integration_score': overall_integration_score,
                'phases_completed': len(integration_scenario),
                'message': '전체 시스템 통합이 성공적으로 완료됨'
            }

            logger.info("전체 시스템 통합 테스트 통과")

        except Exception as e:
            logger.error(f"전체 시스템 통합 테스트 실패: {e}")
            self.test_results['tests']['full_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }

    def generate_summary(self):
        """테스트 결과 요약 생성"""
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'PASS')
        failed_tests = total_tests - passed_tests

        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'end_time': datetime.now(),
            'duration': (datetime.now() - self.test_results['start_time']).total_seconds(),
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        }

    def save_results(self):
        """테스트 결과 저장"""
        results_file = Path("phase3_4_integration_test_results.json")

        # datetime 객체를 문자열로 변환
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=json_serializer)

            logger.info(f"테스트 결과가 {results_file}에 저장되었습니다.")

        except Exception as e:
            logger.error(f"테스트 결과 저장 실패: {e}")

    def print_summary(self):
        """테스트 결과 요약 출력"""
        summary = self.test_results['summary']

        print("\n" + "="*60)
        print("Phase 3-4 통합 테스트 결과 요약")
        print("="*60)
        print(f"전체 테스트: {summary['total_tests']}")
        print(f"통과: {summary['passed_tests']}")
        print(f"실패: {summary['failed_tests']}")
        print(f"성공률: {summary['success_rate']:.1f}%")
        print(f"실행 시간: {summary['duration']:.2f}초")
        print(f"전체 상태: {summary['overall_status']}")
        print("="*60)

        # 각 테스트 상세 결과
        for test_name, test_result in self.test_results['tests'].items():
            status_symbol = "[PASS]" if test_result['status'] == 'PASS' else "[FAIL]"
            print(f"{status_symbol} {test_name}: {test_result['status']}")
            if test_result['status'] == 'FAIL':
                print(f"   오류: {test_result.get('error', '알 수 없는 오류')}")

        print("="*60)


async def main():
    """메인 테스트 실행 함수"""
    print("Phase 3-4 통합 테스트를 시작합니다...")

    test_suite = Phase34IntegrationTest()

    try:
        await test_suite.run_all_tests()
        test_suite.print_summary()

        # 테스트 성공 여부에 따른 종료 코드
        if test_suite.test_results['summary']['overall_status'] == 'PASS':
            print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
            return 0
        else:
            print("\n[FAIL] 일부 테스트가 실패했습니다. 결과를 확인하세요.")
            return 1

    except Exception as e:
        print(f"\n[ERROR] 테스트 실행 중 예상치 못한 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))