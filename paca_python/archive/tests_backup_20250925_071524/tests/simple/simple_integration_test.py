"""
Phase 9.1: 시스템 통합 검증 (간단 버전)
PACA 전체 시스템의 통합 상태를 검증하는 간단한 테스트
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from paca.tools import (
    ReActFramework, PACAToolManager, SafetyPolicy
)
from paca.tools.tools import WebSearchTool, FileManagerTool


class SimpleIntegrationAnalyzer:
    """PACA 시스템의 기본 통합 상태를 분석하는 간단한 클래스"""

    def __init__(self):
        self.components = {}
        self.test_results = []

    async def initialize_core_components(self):
        """핵심 컴포넌트들만 초기화"""
        print("=== PACA 핵심 시스템 초기화 ===")

        try:
            # Phase 8: Tools (확실히 구현된 부분)
            print("1. 도구 시스템 초기화...")
            self.components['tool_manager'] = PACAToolManager()
            self.components['web_search'] = WebSearchTool()
            self.components['file_manager'] = FileManagerTool()
            self.components['react_framework'] = ReActFramework(self.components['tool_manager'])

            # 도구 등록
            register_result1 = self.components['tool_manager'].register_tool(self.components['web_search'])
            register_result2 = self.components['tool_manager'].register_tool(self.components['file_manager'])

            print(f"   웹 검색 도구 등록: {'성공' if register_result1 else '실패'}")
            print(f"   파일 관리 도구 등록: {'성공' if register_result2 else '실패'}")

            print("완료: 모든 핵심 컴포넌트 초기화 완료")
            return True

        except Exception as e:
            print(f"오류: 컴포넌트 초기화 실패 - {e}")
            return False

    async def test_basic_operations(self):
        """기본 동작 테스트"""
        print("\n=== 기본 동작 테스트 ===")

        test_results = []

        try:
            # 1. 도구 관리자 테스트
            print("1. 도구 관리자 기능 테스트...")
            tools = self.components['tool_manager'].list_tools()
            test_results.append({
                'component': 'tool_manager',
                'test': 'list_tools',
                'status': 'success',
                'result': f"{len(tools)}개 도구 발견"
            })

            # 2. 웹 검색 테스트
            print("2. 웹 검색 기능 테스트...")
            search_result = await self.components['tool_manager'].execute_tool(
                'web_search',
                query='Python programming'
            )
            test_results.append({
                'component': 'web_search',
                'test': 'basic_search',
                'status': 'success' if search_result.success else 'failed',
                'result': f"검색 성공: {search_result.success}, 결과: {len(search_result.data) if search_result.data else 0}개"
            })

            # 3. 파일 관리 테스트
            print("3. 파일 관리 기능 테스트...")

            # 테스트 파일 생성
            write_result = await self.components['tool_manager'].execute_tool(
                'file_manager',
                operation='write',
                path='integration_test.txt',
                content='Integration test content'
            )

            # 파일 읽기
            read_result = await self.components['tool_manager'].execute_tool(
                'file_manager',
                operation='read',
                path='integration_test.txt'
            )

            test_results.append({
                'component': 'file_manager',
                'test': 'write_read_file',
                'status': 'success' if write_result.success and read_result.success else 'failed',
                'result': f"파일 쓰기: {write_result.success}, 파일 읽기: {read_result.success}"
            })

            # 4. ReAct 프레임워크 테스트
            print("4. ReAct 프레임워크 테스트...")
            session = await self.components['react_framework'].create_session(
                goal="간단한 통합 테스트",
                max_steps=3
            )

            # 생각 단계
            think_step = await self.components['react_framework'].think(
                session,
                "통합 테스트를 수행하고 있습니다."
            )

            test_results.append({
                'component': 'react_framework',
                'test': 'session_creation',
                'status': 'success',
                'result': f"세션 생성 완료, 단계 수: {len(session.steps)}"
            })

        except Exception as e:
            test_results.append({
                'component': 'integration_test',
                'test': 'error_handling',
                'status': 'error',
                'result': f"오류 발생: {str(e)}"
            })

        return test_results

    async def measure_performance(self):
        """성능 측정"""
        print("\n=== 성능 측정 ===")

        performance_data = {}

        try:
            # 1. 도구 실행 시간 측정
            start_time = time.time()
            result = await self.components['tool_manager'].execute_tool(
                'web_search',
                query='performance test'
            )
            execution_time = time.time() - start_time

            performance_data['web_search_time'] = execution_time
            performance_data['web_search_success'] = result.success

            # 2. 파일 작업 시간 측정
            start_time = time.time()
            await self.components['tool_manager'].execute_tool(
                'file_manager',
                operation='write',
                path='perf_test.txt',
                content='Performance test content'
            )
            file_write_time = time.time() - start_time

            performance_data['file_write_time'] = file_write_time

            # 3. 세션 생성 시간 측정
            start_time = time.time()
            session = await self.components['react_framework'].create_session(
                goal="성능 테스트",
                max_steps=1
            )
            session_time = time.time() - start_time

            performance_data['session_creation_time'] = session_time

        except Exception as e:
            performance_data['error'] = str(e)

        return performance_data

    def analyze_integration_health(self, test_results, performance_data):
        """통합 상태 건강도 분석"""
        print("\n=== 통합 상태 분석 ===")

        analysis = {
            'overall_health': 'unknown',
            'component_status': {},
            'performance_grade': 'unknown',
            'recommendations': []
        }

        # 컴포넌트 상태 분석
        success_count = 0
        total_tests = len(test_results)

        for result in test_results:
            component = result['component']
            status = result['status']

            if component not in analysis['component_status']:
                analysis['component_status'][component] = []

            analysis['component_status'][component].append(status)

            if status == 'success':
                success_count += 1

        # 전체 건강도 계산
        if total_tests > 0:
            success_rate = success_count / total_tests
            if success_rate >= 0.9:
                analysis['overall_health'] = 'excellent'
            elif success_rate >= 0.7:
                analysis['overall_health'] = 'good'
            elif success_rate >= 0.5:
                analysis['overall_health'] = 'fair'
            else:
                analysis['overall_health'] = 'poor'

        # 성능 등급 분석
        if 'error' not in performance_data:
            avg_time = sum([
                performance_data.get('web_search_time', 0),
                performance_data.get('file_write_time', 0),
                performance_data.get('session_creation_time', 0)
            ]) / 3

            if avg_time < 1.0:
                analysis['performance_grade'] = 'excellent'
            elif avg_time < 3.0:
                analysis['performance_grade'] = 'good'
            elif avg_time < 5.0:
                analysis['performance_grade'] = 'fair'
            else:
                analysis['performance_grade'] = 'poor'

        # 권장사항 생성
        if analysis['overall_health'] in ['fair', 'poor']:
            analysis['recommendations'].append("컴포넌트 간 연결성 개선 필요")

        if analysis['performance_grade'] in ['fair', 'poor']:
            analysis['recommendations'].append("성능 최적화 필요")

        return analysis

    async def generate_integration_report(self):
        """통합 보고서 생성"""
        print("\n=== 통합 보고서 생성 ===")

        # 모든 테스트 실행
        test_results = await self.test_basic_operations()
        performance_data = await self.measure_performance()
        health_analysis = self.analyze_integration_health(test_results, performance_data)

        # 보고서 작성
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 9.1 - 시스템 통합 검증',
            'components_tested': list(self.components.keys()),
            'test_summary': {
                'total_tests': len(test_results),
                'successful_tests': len([r for r in test_results if r['status'] == 'success']),
                'failed_tests': len([r for r in test_results if r['status'] == 'failed']),
                'error_tests': len([r for r in test_results if r['status'] == 'error'])
            },
            'test_details': test_results,
            'performance_metrics': performance_data,
            'health_analysis': health_analysis,
            'next_steps': self._generate_next_steps(health_analysis)
        }

        return report

    def _generate_next_steps(self, health_analysis):
        """다음 단계 제안 생성"""
        next_steps = []

        if health_analysis['overall_health'] == 'excellent':
            next_steps.append("Phase 9.2: 성능 프로파일링 진행 준비")
        elif health_analysis['overall_health'] == 'good':
            next_steps.append("미세 조정 후 Phase 9.2 진행")
        else:
            next_steps.append("통합 문제 해결 우선 필요")

        if health_analysis['performance_grade'] in ['fair', 'poor']:
            next_steps.append("성능 병목점 분석 필요")

        if health_analysis['recommendations']:
            next_steps.extend(health_analysis['recommendations'])

        return next_steps


async def main():
    """메인 실행 함수"""
    print("PACA Phase 9.1: 시스템 통합 검증 (간단 버전)")
    print("=" * 60)

    analyzer = SimpleIntegrationAnalyzer()

    try:
        # 1. 시스템 초기화
        if not await analyzer.initialize_core_components():
            print("시스템 초기화 실패")
            return False

        # 2. 통합 보고서 생성
        report = await analyzer.generate_integration_report()

        # 3. 결과 출력
        print(f"\n결과 요약:")
        print(f"   테스트된 컴포넌트: {len(report['components_tested'])}개")
        print(f"   총 테스트: {report['test_summary']['total_tests']}개")
        print(f"   성공: {report['test_summary']['successful_tests']}개")
        print(f"   실패: {report['test_summary']['failed_tests']}개")
        print(f"   오류: {report['test_summary']['error_tests']}개")
        print(f"   전체 상태: {report['health_analysis']['overall_health']}")
        print(f"   성능 등급: {report['health_analysis']['performance_grade']}")

        if report['health_analysis']['recommendations']:
            print(f"\n권장사항:")
            for i, rec in enumerate(report['health_analysis']['recommendations'], 1):
                print(f"   {i}. {rec}")

        if report['next_steps']:
            print(f"\n다음 단계:")
            for i, step in enumerate(report['next_steps'], 1):
                print(f"   {i}. {step}")

        # 4. 보고서 파일 저장
        report_file = "integration_report_simple.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n통합 분석 완료! 상세 보고서: {report_file}")
        return True

    except Exception as e:
        print(f"분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)