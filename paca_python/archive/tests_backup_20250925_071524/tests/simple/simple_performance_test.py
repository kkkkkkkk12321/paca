"""
Phase 9.2: 성능 분석 (간단 버전)
PACA 시스템의 기본 성능 측정 및 최적화 포인트 발견
"""

import asyncio
import sys
import os
import time
import json
import gc
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from paca.tools import (
    ReActFramework, PACAToolManager, SafetyPolicy
)
from paca.tools.tools import WebSearchTool, FileManagerTool


class SimplePerformanceAnalyzer:
    """PACA 시스템의 기본 성능을 분석하는 간단한 클래스"""

    def __init__(self):
        self.components = {}
        self.metrics = {}

    async def setup_components(self):
        """컴포넌트 설정"""
        print("=== 성능 측정 환경 설정 ===")

        try:
            self.components['tool_manager'] = PACAToolManager()
            self.components['web_search'] = WebSearchTool()
            self.components['file_manager'] = FileManagerTool()
            self.components['react_framework'] = ReActFramework(self.components['tool_manager'])

            # 도구 등록
            self.components['tool_manager'].register_tool(self.components['web_search'])
            self.components['tool_manager'].register_tool(self.components['file_manager'])

            print("환경 설정 완료")
            return True

        except Exception as e:
            print(f"환경 설정 실패: {e}")
            return False

    async def measure_basic_operations(self):
        """기본 연산 성능 측정"""
        print("\n=== 기본 연산 성능 측정 ===")

        results = {}

        # 1. 웹 검색 성능 측정
        print("1. 웹 검색 성능 측정...")
        try:
            start_time = time.time()
            result = await self.components['tool_manager'].execute_tool(
                'web_search',
                query='Python performance'
            )
            end_time = time.time()

            results['web_search'] = {
                'execution_time': end_time - start_time,
                'success': result.success,
                'result_count': len(result.data) if result.data else 0
            }
            print(f"   웹 검색: {results['web_search']['execution_time']:.3f}초")

        except Exception as e:
            results['web_search'] = {
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }

        # 2. 파일 작업 성능 측정
        print("2. 파일 작업 성능 측정...")
        try:
            # 파일 쓰기
            start_time = time.time()
            write_result = await self.components['tool_manager'].execute_tool(
                'file_manager',
                operation='write',
                path='performance_test.txt',
                content='Performance test content for PACA system'
            )
            write_time = time.time() - start_time

            # 파일 읽기
            start_time = time.time()
            read_result = await self.components['tool_manager'].execute_tool(
                'file_manager',
                operation='read',
                path='performance_test.txt'
            )
            read_time = time.time() - start_time

            results['file_operations'] = {
                'write_time': write_time,
                'read_time': read_time,
                'write_success': write_result.success,
                'read_success': read_result.success
            }
            print(f"   파일 쓰기: {write_time:.3f}초")
            print(f"   파일 읽기: {read_time:.3f}초")

        except Exception as e:
            results['file_operations'] = {
                'error': str(e)
            }

        # 3. ReAct 프레임워크 성능 측정
        print("3. ReAct 프레임워크 성능 측정...")
        try:
            # 세션 생성
            start_time = time.time()
            session = await self.components['react_framework'].create_session(
                goal="성능 측정 테스트",
                max_steps=5
            )
            session_time = time.time() - start_time

            # 생각 단계
            start_time = time.time()
            await self.components['react_framework'].think(
                session,
                "성능 측정을 위한 생각입니다."
            )
            think_time = time.time() - start_time

            results['react_framework'] = {
                'session_creation_time': session_time,
                'think_time': think_time,
                'total_steps': len(session.steps)
            }
            print(f"   세션 생성: {session_time:.3f}초")
            print(f"   생각 단계: {think_time:.3f}초")

        except Exception as e:
            results['react_framework'] = {
                'error': str(e)
            }

        return results

    async def measure_repeated_operations(self, iterations=3):
        """반복 연산 성능 측정"""
        print(f"\n=== 반복 연산 성능 측정 ({iterations}회) ===")

        repeated_results = {}

        # 웹 검색 반복 테스트
        print("1. 웹 검색 반복 테스트...")
        search_times = []
        search_successes = 0

        for i in range(iterations):
            try:
                start_time = time.time()
                result = await self.components['tool_manager'].execute_tool(
                    'web_search',
                    query=f'test query {i}'
                )
                end_time = time.time()

                search_times.append(end_time - start_time)
                if result.success:
                    search_successes += 1

                # 메모리 정리
                gc.collect()

            except Exception as e:
                print(f"   반복 {i+1} 실패: {e}")

        if search_times:
            repeated_results['web_search_repeated'] = {
                'average_time': sum(search_times) / len(search_times),
                'min_time': min(search_times),
                'max_time': max(search_times),
                'success_rate': search_successes / iterations,
                'iterations': iterations
            }
            print(f"   평균 시간: {repeated_results['web_search_repeated']['average_time']:.3f}초")
            print(f"   성공률: {repeated_results['web_search_repeated']['success_rate']:.1%}")

        # 파일 작업 반복 테스트
        print("2. 파일 작업 반복 테스트...")
        file_times = []
        file_successes = 0

        for i in range(iterations):
            try:
                start_time = time.time()
                result = await self.components['tool_manager'].execute_tool(
                    'file_manager',
                    operation='write',
                    path=f'repeat_test_{i}.txt',
                    content=f'Repeat test content {i}'
                )
                end_time = time.time()

                file_times.append(end_time - start_time)
                if result.success:
                    file_successes += 1

                gc.collect()

            except Exception as e:
                print(f"   반복 {i+1} 실패: {e}")

        if file_times:
            repeated_results['file_operations_repeated'] = {
                'average_time': sum(file_times) / len(file_times),
                'min_time': min(file_times),
                'max_time': max(file_times),
                'success_rate': file_successes / iterations,
                'iterations': iterations
            }
            print(f"   평균 시간: {repeated_results['file_operations_repeated']['average_time']:.3f}초")
            print(f"   성공률: {repeated_results['file_operations_repeated']['success_rate']:.1%}")

        return repeated_results

    def analyze_performance(self, basic_results, repeated_results):
        """성능 분석"""
        print("\n=== 성능 분석 ===")

        analysis = {
            'overall_performance': 'unknown',
            'bottlenecks': [],
            'recommendations': [],
            'optimization_opportunities': []
        }

        # 기본 성능 분석
        slow_operations = []
        fast_operations = []

        # 웹 검색 분석
        if 'web_search' in basic_results and basic_results['web_search'].get('execution_time', 0) > 0:
            time_val = basic_results['web_search']['execution_time']
            if time_val > 1.0:
                slow_operations.append(f"웹 검색 ({time_val:.3f}초)")
                analysis['bottlenecks'].append({
                    'operation': 'web_search',
                    'issue': '응답 시간 지연',
                    'value': f"{time_val:.3f}초",
                    'recommendation': '캐싱 도입, 비동기 최적화 검토'
                })
            else:
                fast_operations.append(f"웹 검색 ({time_val:.3f}초)")

        # 파일 작업 분석
        if 'file_operations' in basic_results:
            file_ops = basic_results['file_operations']
            if 'write_time' in file_ops and file_ops['write_time'] > 0.1:
                slow_operations.append(f"파일 쓰기 ({file_ops['write_time']:.3f}초)")
                analysis['bottlenecks'].append({
                    'operation': 'file_write',
                    'issue': '파일 쓰기 지연',
                    'value': f"{file_ops['write_time']:.3f}초",
                    'recommendation': '디스크 I/O 최적화, 버퍼링 개선'
                })

        # 반복 연산 분석
        if 'web_search_repeated' in repeated_results:
            repeated = repeated_results['web_search_repeated']
            if repeated['success_rate'] < 0.9:
                analysis['bottlenecks'].append({
                    'operation': 'web_search_reliability',
                    'issue': '낮은 성공률',
                    'value': f"{repeated['success_rate']:.1%}",
                    'recommendation': '에러 처리 개선, 재시도 로직 추가'
                })

        # 전체 성능 등급 결정
        if len(slow_operations) == 0:
            analysis['overall_performance'] = 'excellent'
        elif len(slow_operations) <= 2:
            analysis['overall_performance'] = 'good'
        else:
            analysis['overall_performance'] = 'needs_improvement'

        # 최적화 기회 식별
        if len(analysis['bottlenecks']) > 0:
            analysis['optimization_opportunities'] = [
                '비동기 처리 최적화',
                '캐싱 레이어 도입',
                '에러 처리 강화',
                '리소스 풀링 구현'
            ]

        # 권장사항 생성
        if analysis['overall_performance'] == 'excellent':
            analysis['recommendations'] = [
                'Phase 9.3 메모리 최적화 진행 준비',
                '현재 성능 수준 유지'
            ]
        elif analysis['overall_performance'] == 'good':
            analysis['recommendations'] = [
                '미세 조정 후 다음 단계 진행',
                '모니터링 강화'
            ]
        else:
            analysis['recommendations'] = [
                '성능 병목점 우선 해결 필요',
                '아키텍처 리뷰 고려'
            ]

        print(f"   전체 성능: {analysis['overall_performance']}")
        print(f"   병목점: {len(analysis['bottlenecks'])}개 발견")

        if analysis['bottlenecks']:
            print("   주요 병목점:")
            for bottleneck in analysis['bottlenecks']:
                print(f"     - {bottleneck['operation']}: {bottleneck['issue']} ({bottleneck['value']})")

        return analysis

    async def generate_performance_report(self):
        """성능 보고서 생성"""
        print("\n=== 성능 보고서 생성 ===")

        # 모든 측정 실행
        basic_results = await self.measure_basic_operations()
        repeated_results = await self.measure_repeated_operations()
        analysis = self.analyze_performance(basic_results, repeated_results)

        # 보고서 작성
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'Phase 9.2 - 성능 프로파일링 및 병목점 해결',
            'basic_performance': basic_results,
            'repeated_performance': repeated_results,
            'performance_analysis': analysis,
            'summary': {
                'overall_grade': analysis['overall_performance'],
                'bottlenecks_count': len(analysis['bottlenecks']),
                'optimization_opportunities': len(analysis['optimization_opportunities']),
                'next_phase_ready': analysis['overall_performance'] in ['excellent', 'good']
            },
            'next_steps': analysis['recommendations']
        }

        return report


async def main():
    """메인 실행 함수"""
    print("PACA Phase 9.2: 성능 분석 (간단 버전)")
    print("=" * 60)

    analyzer = SimplePerformanceAnalyzer()

    try:
        # 1. 환경 설정
        if not await analyzer.setup_components():
            print("환경 설정 실패")
            return False

        # 2. 성능 보고서 생성
        report = await analyzer.generate_performance_report()

        # 3. 결과 출력
        print(f"\n성능 분석 결과:")
        print(f"   전체 등급: {report['summary']['overall_grade']}")
        print(f"   병목점: {report['summary']['bottlenecks_count']}개")
        print(f"   최적화 기회: {report['summary']['optimization_opportunities']}개")
        print(f"   다음 단계 준비: {'준비됨' if report['summary']['next_phase_ready'] else '개선 필요'}")

        if report['performance_analysis']['bottlenecks']:
            print(f"\n주요 병목점:")
            for i, bottleneck in enumerate(report['performance_analysis']['bottlenecks'], 1):
                print(f"   {i}. {bottleneck['operation']}: {bottleneck['issue']}")
                print(f"      권장사항: {bottleneck['recommendation']}")

        if report['next_steps']:
            print(f"\n다음 단계:")
            for i, step in enumerate(report['next_steps'], 1):
                print(f"   {i}. {step}")

        # 4. 보고서 파일 저장
        report_file = "performance_report_simple.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n성능 분석 완료! 상세 보고서: {report_file}")
        return True

    except Exception as e:
        print(f"성능 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)