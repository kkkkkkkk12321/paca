"""
Phase 9.1: 시스템 통합 검증 및 최적화
PACA 전체 시스템의 통합 상태를 검증하고 최적화 포인트를 찾는 테스트
"""

import asyncio
import sys
import os
import time
import psutil
import tracemalloc
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from paca.cognitive import (
    CognitiveSystem, ComplexityDetector, MetacognitionEngine,
    ReasoningChain, WorkingMemory, LongTermMemory
)
try:
    from paca.governance import EthicalFramework, SafetyMonitor
except ImportError:
    print("거버넌스 모듈을 찾을 수 없습니다. 기본 구현을 사용합니다.")
    EthicalFramework = None
    SafetyMonitor = None

try:
    from paca.truth_seeking import TruthSeeker
except ImportError:
    print("진실 탐구 모듈을 찾을 수 없습니다. 기본 구현을 사용합니다.")
    TruthSeeker = None
from paca.tools import (
    ReActFramework, PACAToolManager, SafetyPolicy
)
from paca.tools.tools import WebSearchTool, FileManagerTool


class SystemIntegrationAnalyzer:
    """PACA 시스템 전체의 통합 상태를 분석하고 최적화 포인트를 찾는 클래스"""

    def __init__(self):
        self.components = {}
        self.performance_metrics = {}
        self.integration_issues = []
        self.optimization_recommendations = []

    async def initialize_all_components(self):
        """모든 PACA 컴포넌트를 초기화"""
        print("=== PACA 전체 시스템 초기화 ===")

        try:
            # Phase 5: Cognitive Components
            print("1. 인지 시스템 초기화...")
            self.components['cognitive_system'] = CognitiveSystem()
            self.components['complexity_detector'] = ComplexityDetector()
            self.components['metacognition_engine'] = MetacognitionEngine()
            self.components['reasoning_chain'] = ReasoningChain()
            self.components['working_memory'] = WorkingMemory()
            self.components['longterm_memory'] = LongTermMemory()

            # Phase 6: Governance
            print("2. 거버넌스 시스템 초기화...")
            if EthicalFramework:
                self.components['ethical_framework'] = EthicalFramework()
            if SafetyMonitor:
                self.components['safety_monitor'] = SafetyMonitor()

            # Phase 7: Truth Seeking
            print("3. 진실 탐구 시스템 초기화...")
            if TruthSeeker:
                self.components['truth_seeker'] = TruthSeeker()

            # Phase 8: Tools
            print("4. 도구 시스템 초기화...")
            self.components['tool_manager'] = PACAToolManager()
            self.components['web_search'] = WebSearchTool()
            self.components['file_manager'] = FileManagerTool()
            self.components['react_framework'] = ReActFramework(self.components['tool_manager'])

            # 도구 등록
            self.components['tool_manager'].register_tool(self.components['web_search'])
            self.components['tool_manager'].register_tool(self.components['file_manager'])

            print("✅ 모든 컴포넌트 초기화 완료")
            return True

        except Exception as e:
            print(f"❌ 컴포넌트 초기화 실패: {e}")
            traceback.print_exc()
            return False

    async def analyze_component_integration(self):
        """컴포넌트 간 통합 상태 분석"""
        print("\n=== 컴포넌트 통합 분석 ===")

        integration_matrix = {}

        # 각 컴포넌트의 의존성 및 연결성 확인
        for name, component in self.components.items():
            integration_info = {
                'status': 'active' if component else 'inactive',
                'methods': [method for method in dir(component) if not method.startswith('_')],
                'dependencies': self._analyze_dependencies(component),
                'interfaces': self._analyze_interfaces(component)
            }
            integration_matrix[name] = integration_info

        return integration_matrix

    def _analyze_dependencies(self, component):
        """컴포넌트의 의존성 분석"""
        dependencies = []

        # 컴포넌트의 속성에서 다른 PACA 컴포넌트 참조 찾기
        for attr_name in dir(component):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(component, attr_name)
                    if hasattr(attr_value, '__module__') and 'paca' in str(attr_value.__module__):
                        dependencies.append(attr_name)
                except:
                    pass

        return dependencies

    def _analyze_interfaces(self, component):
        """컴포넌트의 인터페이스 분석"""
        interfaces = []

        for method_name in dir(component):
            if not method_name.startswith('_') and callable(getattr(component, method_name, None)):
                method = getattr(component, method_name)
                if hasattr(method, '__annotations__'):
                    interfaces.append({
                        'method': method_name,
                        'annotations': str(method.__annotations__)
                    })

        return interfaces

    async def test_cross_component_communication(self):
        """크로스 컴포넌트 통신 테스트"""
        print("\n=== 크로스 컴포넌트 통신 테스트 ===")

        communication_tests = []

        try:
            # 1. Tools → Memory 통합 테스트
            print("1. 도구 시스템과 메모리 통합 테스트...")

            # ReAct 세션 생성
            session = await self.components['react_framework'].create_session(
                goal="시스템 통합 테스트",
                max_steps=5
            )

            # 웹 검색 실행
            search_result = await self.components['tool_manager'].execute_tool(
                'web_search',
                query='AI system integration'
            )

            # 메모리에 결과 저장 (메모리 시스템이 있다면)
            if hasattr(self.components['working_memory'], 'store'):
                memory_result = self.components['working_memory'].store(
                    'search_result',
                    search_result.data if search_result.success else None
                )
                communication_tests.append({
                    'test': 'Tools → Memory',
                    'status': 'success' if memory_result else 'failed',
                    'details': f"Search success: {search_result.success}, Memory storage: {bool(memory_result)}"
                })
            else:
                communication_tests.append({
                    'test': 'Tools → Memory',
                    'status': 'partial',
                    'details': f"Search success: {search_result.success}, Memory interface not available"
                })

            # 2. Ethics → Tools 통합 테스트
            print("2. 윤리 시스템과 도구 통합 테스트...")

            # 안전성 정책 적용 테스트
            if hasattr(self.components['ethical_framework'], 'evaluate_action'):
                ethics_evaluation = self.components['ethical_framework'].evaluate_action(
                    action_type='web_search',
                    parameters={'query': 'test query'}
                )
                communication_tests.append({
                    'test': 'Ethics → Tools',
                    'status': 'success',
                    'details': f"Ethics evaluation: {ethics_evaluation}"
                })

            # 3. Cognitive → All 통합 테스트
            print("3. 인지 시스템 통합 테스트...")

            # 복잡도 감지 테스트
            if hasattr(self.components['complexity_detector'], 'detect'):
                complexity_result = self.components['complexity_detector'].detect(
                    task_type='web_search',
                    context={'query': 'AI system integration'}
                )
                communication_tests.append({
                    'test': 'Complexity Detection',
                    'status': 'success',
                    'details': f"Complexity analysis: {complexity_result}"
                })

            # 메타인지 엔진 테스트
            if hasattr(self.components['metacognition_engine'], 'monitor'):
                meta_result = self.components['metacognition_engine'].monitor(
                    process_type='tool_execution',
                    data={'tool': 'web_search', 'success': search_result.success}
                )
                communication_tests.append({
                    'test': 'Metacognition Integration',
                    'status': 'success',
                    'details': f"Metacognition result: {meta_result}"
                })

        except Exception as e:
            communication_tests.append({
                'test': 'Cross-component Communication',
                'status': 'error',
                'details': f"Error: {str(e)}"
            })

        return communication_tests

    async def performance_profiling(self):
        """시스템 성능 프로파일링"""
        print("\n=== 성능 프로파일링 ===")

        # 메모리 추적 시작
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        performance_data = {
            'startup_time': 0,
            'memory_usage': {},
            'execution_times': {},
            'resource_usage': {}
        }

        # 시작 시간 측정
        start_time = time.time()

        # 각 컴포넌트의 기본 작업 성능 측정
        for name, component in self.components.items():
            if name == 'tool_manager':
                # 도구 실행 성능 측정
                exec_start = time.time()
                result = await component.execute_tool('web_search', query='test')
                exec_time = time.time() - exec_start
                performance_data['execution_times'][name] = exec_time

            elif name == 'react_framework':
                # ReAct 세션 생성 성능 측정
                exec_start = time.time()
                session = await component.create_session(goal="performance test", max_steps=1)
                exec_time = time.time() - exec_start
                performance_data['execution_times'][name] = exec_time

        # 전체 실행 시간
        total_time = time.time() - start_time
        performance_data['startup_time'] = total_time

        # 메모리 사용량
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        performance_data['memory_usage'] = {
            'initial': start_memory,
            'current': current_memory,
            'increase': current_memory - start_memory
        }

        # 리소스 사용량
        cpu_percent = psutil.cpu_percent(interval=1)
        performance_data['resource_usage'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': psutil.virtual_memory().percent
        }

        # 메모리 추적 중지
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        performance_data['memory_usage']['traced_current'] = current / 1024 / 1024  # MB
        performance_data['memory_usage']['traced_peak'] = peak / 1024 / 1024  # MB

        return performance_data

    def identify_optimization_opportunities(self, integration_matrix, communication_tests, performance_data):
        """최적화 기회 식별"""
        print("\n=== 최적화 기회 식별 ===")

        opportunities = []

        # 1. 성능 기반 최적화
        if performance_data['memory_usage']['increase'] > 100:  # 100MB 이상 증가
            opportunities.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'description': f"메모리 사용량이 {performance_data['memory_usage']['increase']:.1f}MB 증가함",
                'recommendation': "메모리 풀링, 객체 재사용, 가비지 컬렉션 최적화 필요"
            })

        if any(time > 1.0 for time in performance_data['execution_times'].values()):  # 1초 이상 실행
            slow_components = [name for name, time in performance_data['execution_times'].items() if time > 1.0]
            opportunities.append({
                'type': 'performance_optimization',
                'priority': 'medium',
                'description': f"느린 컴포넌트: {', '.join(slow_components)}",
                'recommendation': "비동기 처리 개선, 캐싱 전략 적용 필요"
            })

        # 2. 통합 기반 최적화
        disconnected_components = []
        for name, info in integration_matrix.items():
            if not info['dependencies'] and name not in ['tool_manager', 'react_framework']:
                disconnected_components.append(name)

        if disconnected_components:
            opportunities.append({
                'type': 'integration_optimization',
                'priority': 'medium',
                'description': f"연결되지 않은 컴포넌트: {', '.join(disconnected_components)}",
                'recommendation': "컴포넌트 간 인터페이스 개선, 이벤트 시스템 도입 필요"
            })

        # 3. 통신 기반 최적화
        failed_tests = [test for test in communication_tests if test['status'] == 'failed']
        if failed_tests:
            opportunities.append({
                'type': 'communication_optimization',
                'priority': 'high',
                'description': f"실패한 통신 테스트: {len(failed_tests)}개",
                'recommendation': "컴포넌트 간 인터페이스 표준화, 에러 처리 개선 필요"
            })

        return opportunities

    async def generate_integration_report(self):
        """통합 분석 보고서 생성"""
        print("\n=== 통합 분석 보고서 생성 ===")

        # 모든 분석 실행
        integration_matrix = await self.analyze_component_integration()
        communication_tests = await self.test_cross_component_communication()
        performance_data = await self.performance_profiling()
        optimization_opportunities = self.identify_optimization_opportunities(
            integration_matrix, communication_tests, performance_data
        )

        # 보고서 생성
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_status': 'operational',
            'components': {
                'total': len(self.components),
                'active': len([c for c in self.components.values() if c]),
                'integration_score': self._calculate_integration_score(integration_matrix)
            },
            'performance': performance_data,
            'communication': {
                'tests_run': len(communication_tests),
                'success_rate': len([t for t in communication_tests if t['status'] == 'success']) / len(communication_tests) if communication_tests else 0
            },
            'optimization': {
                'opportunities_found': len(optimization_opportunities),
                'high_priority': len([o for o in optimization_opportunities if o['priority'] == 'high']),
                'recommendations': optimization_opportunities
            }
        }

        return report

    def _calculate_integration_score(self, integration_matrix):
        """통합 점수 계산"""
        if not integration_matrix:
            return 0

        total_score = 0
        for name, info in integration_matrix.items():
            component_score = 0

            # 활성 상태 점수
            if info['status'] == 'active':
                component_score += 25

            # 인터페이스 점수
            if info['interfaces']:
                component_score += 25

            # 의존성 점수 (적절한 의존성이 있으면 좋음)
            if info['dependencies']:
                component_score += 25

            # 메소드 점수
            if len(info['methods']) > 5:
                component_score += 25
            elif len(info['methods']) > 0:
                component_score += 10

            total_score += component_score

        return total_score / len(integration_matrix)


async def main():
    """메인 실행 함수"""
    print("PACA Phase 9.1: 시스템 통합 검증 및 최적화")
    print("=" * 60)

    analyzer = SystemIntegrationAnalyzer()

    try:
        # 1. 시스템 초기화
        if not await analyzer.initialize_all_components():
            print("❌ 시스템 초기화 실패")
            return False

        # 2. 통합 분석 보고서 생성
        report = await analyzer.generate_integration_report()

        # 3. 결과 출력
        print(f"\n📊 시스템 통합 분석 결과")
        print(f"   • 총 컴포넌트: {report['components']['total']}개")
        print(f"   • 활성 컴포넌트: {report['components']['active']}개")
        print(f"   • 통합 점수: {report['components']['integration_score']:.1f}/100")
        print(f"   • 통신 성공률: {report['communication']['success_rate']:.1%}")
        print(f"   • 메모리 사용량: {report['performance']['memory_usage']['increase']:.1f}MB 증가")
        print(f"   • 최적화 기회: {report['optimization']['opportunities_found']}개 발견")

        if report['optimization']['high_priority'] > 0:
            print(f"   ⚠️  고우선순위 최적화: {report['optimization']['high_priority']}개")

        # 4. 최적화 권장사항 출력
        if report['optimization']['recommendations']:
            print(f"\n🔧 최적화 권장사항:")
            for i, rec in enumerate(report['optimization']['recommendations'], 1):
                print(f"   {i}. [{rec['priority'].upper()}] {rec['description']}")
                print(f"      → {rec['recommendation']}")

        # 5. 보고서 파일 저장
        report_file = "system_integration_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 통합 분석 완료! 상세 보고서: {report_file}")
        return True

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)