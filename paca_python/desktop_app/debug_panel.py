"""
Debug Panel Module - PACA Python v5
AI의 내부 사고 과정 실시간 표시 및 디버깅 도구

메타인지 엔진과 추론 체인의 내부 동작을 시각화하고
복잡도 분석 결과를 상세하게 표시하는 디버그 인터페이스
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import asyncio
from enum import Enum

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain
    from ..paca.cognitive.complexity_detector import ComplexityResult
    from ..paca.cognitive.reasoning_chain import ReasoningResult, ReasoningStep
    from ..paca.cognitive.metacognition_engine import QualityMetrics
    from ..paca.core.types.base import Result
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain
    from paca.cognitive.complexity_detector import ComplexityResult
    from paca.cognitive.reasoning_chain import ReasoningResult, ReasoningStep
    from paca.cognitive.metacognition_engine import QualityMetrics
    from paca.core.types.base import Result


class DebugLevel(Enum):
    """디버그 레벨"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class DebugEntry:
    """디버그 엔트리"""
    timestamp: datetime
    level: DebugLevel
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None


class ReasoningDisplay:
    """추론 과정 실시간 표시 위젯"""

    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self.reasoning_history: List[ReasoningResult] = []
        self._setup_ui()

    def _setup_ui(self):
        """추론 표시 UI 구성"""
        # 제목
        title_label = tk.Label(
            self.parent_frame,
            text="추론 체인 분석",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 현재 추론 프레임
        current_frame = tk.LabelFrame(
            self.parent_frame,
            text="현재 추론 과정",
            font=('Arial', 12, 'bold')
        )
        current_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 추론 단계 트리뷰
        columns = ('단계', '유형', '설명', '신뢰도', '시간')
        self.reasoning_tree = ttk.Treeview(
            current_frame,
            columns=columns,
            show='headings',
            height=10
        )

        for col in columns:
            self.reasoning_tree.heading(col, text=col)
            width_map = {'단계': 60, '유형': 100, '설명': 300, '신뢰도': 80, '시간': 80}
            self.reasoning_tree.column(col, width=width_map.get(col, 100))

        # 스크롤바
        reasoning_scrollbar = ttk.Scrollbar(
            current_frame,
            orient=tk.VERTICAL,
            command=self.reasoning_tree.yview
        )
        self.reasoning_tree.configure(yscrollcommand=reasoning_scrollbar.set)

        self.reasoning_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        reasoning_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # 상세 정보 프레임
        detail_frame = tk.LabelFrame(
            self.parent_frame,
            text="선택된 단계 상세 정보",
            font=('Arial', 11, 'bold')
        )
        detail_frame.pack(fill=tk.X, padx=10, pady=5)

        self.detail_display = scrolledtext.ScrolledText(
            detail_frame,
            height=6,
            font=('Consolas', 9),
            state=tk.DISABLED
        )
        self.detail_display.pack(fill=tk.X, padx=5, pady=5)

        # 이벤트 바인딩
        self.reasoning_tree.bind('<<TreeviewSelect>>', self._on_step_select)

    def show_reasoning_steps(self, reasoning_chain: List[ReasoningStep]):
        """추론 단계들을 표시"""
        # 기존 항목 제거
        for item in self.reasoning_tree.get_children():
            self.reasoning_tree.delete(item)

        # 새 단계들 추가
        for i, step in enumerate(reasoning_chain, 1):
            self.reasoning_tree.insert('', tk.END, values=(
                str(i),
                step.step_type.value if hasattr(step, 'step_type') else 'Unknown',
                step.description[:50] + '...' if len(step.description) > 50 else step.description,
                f"{step.confidence:.2f}" if hasattr(step, 'confidence') else 'N/A',
                f"{step.duration_ms:.1f}ms" if hasattr(step, 'duration_ms') else 'N/A'
            ))

    def _on_step_select(self, event):
        """추론 단계 선택 이벤트"""
        selection = self.reasoning_tree.selection()
        if not selection:
            return

        item = self.reasoning_tree.item(selection[0])
        step_index = int(item['values'][0]) - 1

        # 상세 정보 표시 (시뮬레이션)
        detail_info = {
            "단계": step_index + 1,
            "설명": item['values'][2],
            "유형": item['values'][1],
            "신뢰도": item['values'][3],
            "처리 시간": item['values'][4],
            "입력 데이터": "사용자 질문 분석 중...",
            "중간 결과": "패턴 매칭 완료",
            "출력 데이터": "다음 단계로 전달",
            "메타데이터": {
                "알고리즘": "순차적 추론",
                "복잡도": "중간",
                "리소스 사용량": "15%"
            }
        }

        self._update_detail_display(detail_info)

    def _update_detail_display(self, detail_info: Dict[str, Any]):
        """상세 정보 업데이트"""
        self.detail_display.config(state=tk.NORMAL)
        self.detail_display.delete(1.0, tk.END)

        # JSON 형태로 예쁘게 출력
        formatted_info = json.dumps(detail_info, indent=2, ensure_ascii=False)
        self.detail_display.insert(tk.END, formatted_info)

        self.detail_display.config(state=tk.DISABLED)

    def add_reasoning_result(self, result: ReasoningResult):
        """추론 결과 추가"""
        self.reasoning_history.append(result)
        if hasattr(result, 'steps'):
            self.show_reasoning_steps(result.steps)


class ComplexityAnalyzer:
    """복잡도 분석 결과 표시 위젯"""

    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self.analysis_history: List[ComplexityResult] = []
        self._setup_ui()

    def _setup_ui(self):
        """복잡도 분석 UI 구성"""
        # 제목
        title_label = tk.Label(
            self.parent_frame,
            text="복잡도 분석",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 현재 분석 결과 프레임
        current_frame = tk.LabelFrame(
            self.parent_frame,
            text="현재 분석 결과",
            font=('Arial', 12, 'bold')
        )
        current_frame.pack(fill=tk.X, padx=10, pady=5)

        # 복잡도 점수 표시
        score_frame = tk.Frame(current_frame)
        score_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(score_frame, text="복잡도 점수:", font=('Arial', 11)).pack(side=tk.LEFT)

        self.score_var = tk.StringVar(value="0")
        self.score_label = tk.Label(
            score_frame,
            textvariable=self.score_var,
            font=('Arial', 14, 'bold'),
            fg='#2196F3'
        )
        self.score_label.pack(side=tk.LEFT, padx=10)

        # 복잡도 막대 그래프
        self.complexity_bar = ttk.Progressbar(
            score_frame,
            length=200,
            mode='determinate'
        )
        self.complexity_bar.pack(side=tk.LEFT, padx=10)

        # 도메인 분석 프레임
        domain_frame = tk.LabelFrame(
            self.parent_frame,
            text="도메인별 분석",
            font=('Arial', 11, 'bold')
        )
        domain_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 도메인 분석 트리뷰
        domain_columns = ('도메인', '점수', '기여도', '키워드')
        self.domain_tree = ttk.Treeview(
            domain_frame,
            columns=domain_columns,
            show='headings',
            height=8
        )

        for col in domain_columns:
            self.domain_tree.heading(col, text=col)
            width_map = {'도메인': 100, '점수': 80, '기여도': 80, '키워드': 200}
            self.domain_tree.column(col, width=width_map.get(col, 100))

        self.domain_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 분석 통계 프레임
        stats_frame = tk.LabelFrame(
            self.parent_frame,
            text="분석 통계",
            font=('Arial', 11, 'bold')
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_display = scrolledtext.ScrolledText(
            stats_frame,
            height=4,
            font=('Arial', 9),
            state=tk.DISABLED
        )
        self.stats_display.pack(fill=tk.X, padx=5, pady=5)

    def display_complexity_analysis(self, analysis: ComplexityResult):
        """복잡도 분석 결과 표시"""
        try:
            # 점수 업데이트
            self.score_var.set(f"{analysis.score}/100")
            self.complexity_bar['value'] = analysis.score

            # 점수에 따른 색상 변경
            if analysis.score < 30:
                color = '#4CAF50'  # 녹색 (간단)
            elif analysis.score < 70:
                color = '#FF9800'  # 주황색 (중간)
            else:
                color = '#f44336'  # 빨간색 (복잡)

            self.score_label.config(fg=color)

            # 도메인 분석 업데이트
            for item in self.domain_tree.get_children():
                self.domain_tree.delete(item)

            # 시뮬레이션 도메인 데이터
            domain_data = [
                ('수학', analysis.score * 0.3, '30%', '계산, 수식, 논리'),
                ('언어', analysis.score * 0.4, '40%', '문법, 어휘, 구조'),
                ('추론', analysis.score * 0.2, '20%', '논리, 인과관계'),
                ('창작', analysis.score * 0.1, '10%', '상상, 아이디어')
            ]

            for domain, score, contribution, keywords in domain_data:
                self.domain_tree.insert('', tk.END, values=(
                    domain,
                    f"{score:.1f}",
                    contribution,
                    keywords
                ))

            # 통계 정보 업데이트
            stats_info = f"""
분석 시간: {datetime.now().strftime('%H:%M:%S')}
추론 필요성: {'예' if analysis.reasoning_required else '아니오'}
도메인: {analysis.domain}
신뢰도: {analysis.confidence:.2f}
키워드 수: {len(analysis.extracted_keywords) if hasattr(analysis, 'extracted_keywords') else 'N/A'}
            """.strip()

            self.stats_display.config(state=tk.NORMAL)
            self.stats_display.delete(1.0, tk.END)
            self.stats_display.insert(tk.END, stats_info)
            self.stats_display.config(state=tk.DISABLED)

            # 히스토리에 추가
            self.analysis_history.append(analysis)

        except Exception as e:
            print(f"복잡도 분석 표시 오류: {e}")

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """분석 통계 조회"""
        if not self.analysis_history:
            return {}

        scores = [result.score for result in self.analysis_history]
        return {
            'total_analyses': len(self.analysis_history),
            'average_complexity': sum(scores) / len(scores),
            'max_complexity': max(scores),
            'min_complexity': min(scores),
            'reasoning_required_count': sum(1 for r in self.analysis_history if r.reasoning_required)
        }


class DebugPanel:
    """
    AI 디버그 패널 메인 클래스

    AI의 내부 사고 과정을 실시간으로 모니터링하고
    복잡도 분석, 추론 체인, 메타인지 과정을 시각화
    """

    def __init__(self, standalone: bool = False):
        """
        디버그 패널 초기화

        Args:
            standalone: 독립 실행 모드 여부
        """
        self.standalone = standalone
        self.debug_entries: List[DebugEntry] = []

        # AI 시스템 초기화
        self.complexity_detector = ComplexityDetector()
        self.reasoning_chain = ReasoningChain()
        self.metacognition = MetacognitionEngine()

        # UI 구성 요소
        self.reasoning_display: Optional[ReasoningDisplay] = None
        self.complexity_analyzer: Optional[ComplexityAnalyzer] = None

        if standalone:
            self.root = tk.Tk()
            self.root.title("PACA v5 Debug Panel")
            self.root.geometry("1000x700")
            self._setup_standalone_ui()
        else:
            self.root = None

    def _setup_standalone_ui(self):
        """독립 실행 모드 UI 구성"""
        if not self.root:
            return

        # 메뉴 바
        self._create_menu()

        # 메인 프레임
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 탭 노트북
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 추론 탭
        reasoning_frame = tk.Frame(self.notebook)
        self.notebook.add(reasoning_frame, text="🧠 추론 과정")
        self.reasoning_display = ReasoningDisplay(reasoning_frame)

        # 복잡도 탭
        complexity_frame = tk.Frame(self.notebook)
        self.notebook.add(complexity_frame, text="📊 복잡도 분석")
        self.complexity_analyzer = ComplexityAnalyzer(complexity_frame)

        # 디버그 로그 탭
        debug_frame = tk.Frame(self.notebook)
        self.notebook.add(debug_frame, text="🔍 디버그 로그")
        self._setup_debug_log_tab(debug_frame)

        # 테스트 탭
        test_frame = tk.Frame(self.notebook)
        self.notebook.add(test_frame, text="🧪 테스트")
        self._setup_test_tab(test_frame)

        # 상태 바
        self._create_status_bar()

    def setup_embedded_ui(self, parent_frame: tk.Frame):
        """임베디드 모드 UI 구성"""
        # 탭 노트북
        self.notebook = ttk.Notebook(parent_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 추론 탭
        reasoning_frame = tk.Frame(self.notebook)
        self.notebook.add(reasoning_frame, text="추론 과정")
        self.reasoning_display = ReasoningDisplay(reasoning_frame)

        # 복잡도 탭
        complexity_frame = tk.Frame(self.notebook)
        self.notebook.add(complexity_frame, text="복잡도 분석")
        self.complexity_analyzer = ComplexityAnalyzer(complexity_frame)

    def _create_menu(self):
        """메뉴 바 생성"""
        if not self.root:
            return

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 디버그 메뉴
        debug_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="디버그", menu=debug_menu)
        debug_menu.add_command(label="로그 지우기", command=self._clear_debug_log)
        debug_menu.add_command(label="테스트 실행", command=self._run_test)
        debug_menu.add_separator()
        debug_menu.add_command(label="설정", command=self._open_debug_settings)

        # 도구 메뉴
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="복잡도 테스트", command=self._test_complexity)
        tools_menu.add_command(label="추론 테스트", command=self._test_reasoning)
        tools_menu.add_command(label="통계 내보내기", command=self._export_statistics)

    def _setup_debug_log_tab(self, parent_frame: tk.Frame):
        """디버그 로그 탭 구성"""
        # 제목
        title_label = tk.Label(
            parent_frame,
            text="디버그 로그",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 로그 레벨 필터
        filter_frame = tk.Frame(parent_frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(filter_frame, text="로그 레벨:").pack(side=tk.LEFT)

        self.log_level_var = tk.StringVar(value="ALL")
        level_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.log_level_var,
            values=["ALL", "TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=10
        )
        level_combo.pack(side=tk.LEFT, padx=5)
        level_combo.bind('<<ComboboxSelected>>', self._filter_debug_log)

        # 로그 표시 영역
        self.debug_log_display = scrolledtext.ScrolledText(
            parent_frame,
            font=('Consolas', 9),
            state=tk.DISABLED
        )
        self.debug_log_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 제어 버튼
        control_frame = tk.Frame(parent_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            control_frame,
            text="로그 지우기",
            command=self._clear_debug_log
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="새로고침",
            command=self._refresh_debug_log
        ).pack(side=tk.LEFT, padx=5)

    def _setup_test_tab(self, parent_frame: tk.Frame):
        """테스트 탭 구성"""
        # 제목
        title_label = tk.Label(
            parent_frame,
            text="기능 테스트",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 입력 프레임
        input_frame = tk.LabelFrame(parent_frame, text="테스트 입력", font=('Arial', 11, 'bold'))
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(input_frame, text="테스트 문장:").pack(anchor=tk.W, padx=10, pady=5)

        self.test_input = tk.Text(input_frame, height=3, font=('Arial', 10))
        self.test_input.pack(fill=tk.X, padx=10, pady=5)
        self.test_input.insert(tk.END, "복잡한 수학 문제를 해결하는 방법을 알려주세요.")

        # 테스트 버튼
        button_frame = tk.Frame(input_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            button_frame,
            text="복잡도 분석",
            command=self._test_complexity,
            bg='#2196F3',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="추론 체인",
            command=self._test_reasoning,
            bg='#4CAF50',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="종합 테스트",
            command=self._run_comprehensive_test,
            bg='#FF9800',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)

        # 결과 표시 영역
        result_frame = tk.LabelFrame(parent_frame, text="테스트 결과", font=('Arial', 11, 'bold'))
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.test_result_display = scrolledtext.ScrolledText(
            result_frame,
            font=('Consolas', 9),
            state=tk.DISABLED
        )
        self.test_result_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_status_bar(self):
        """상태 바 생성"""
        if not self.root:
            return

        self.status_bar = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            self.status_bar,
            text="PACA v5 Debug Panel 준비됨",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def add_debug_entry(self, level: DebugLevel, component: str, message: str, data: Optional[Dict[str, Any]] = None):
        """디버그 엔트리 추가"""
        entry = DebugEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            data=data
        )
        self.debug_entries.append(entry)
        self._update_debug_log_display(entry)

    def _update_debug_log_display(self, entry: DebugEntry):
        """디버그 로그 표시 업데이트"""
        if not hasattr(self, 'debug_log_display'):
            return

        # 레벨 필터링
        if (self.log_level_var.get() != "ALL" and
            self.log_level_var.get() != entry.level.value):
            return

        self.debug_log_display.config(state=tk.NORMAL)

        # 로그 엔트리 포맷
        timestamp_str = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp_str}] {entry.level.value:8} {entry.component:15} {entry.message}\n"

        # 레벨별 색상
        color_map = {
            DebugLevel.TRACE: '#666666',
            DebugLevel.DEBUG: '#0066CC',
            DebugLevel.INFO: '#000000',
            DebugLevel.WARNING: '#FF9800',
            DebugLevel.ERROR: '#f44336'
        }

        # 텍스트 삽입 및 색상 적용
        start_index = self.debug_log_display.index(tk.END)
        self.debug_log_display.insert(tk.END, log_line)
        end_index = self.debug_log_display.index(tk.END)

        tag_name = f"level_{entry.level.value}"
        self.debug_log_display.tag_add(tag_name, start_index, end_index)
        self.debug_log_display.tag_config(tag_name, foreground=color_map.get(entry.level, '#000000'))

        # 데이터가 있으면 추가 표시
        if entry.data:
            data_str = json.dumps(entry.data, indent=2, ensure_ascii=False)
            self.debug_log_display.insert(tk.END, f"    Data: {data_str}\n")

        # 자동 스크롤
        self.debug_log_display.see(tk.END)
        self.debug_log_display.config(state=tk.DISABLED)

    def _test_complexity(self):
        """복잡도 분석 테스트"""
        def run_test():
            try:
                test_text = self.test_input.get(1.0, tk.END).strip()
                if not test_text:
                    return

                self.add_debug_entry(DebugLevel.INFO, "TEST", f"복잡도 분석 시작: {test_text[:30]}...")

                # 비동기 복잡도 분석
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    self.complexity_detector.detect_complexity(test_text)
                )

                if result.is_success:
                    complexity_result = result.value
                    self.add_debug_entry(
                        DebugLevel.INFO,
                        "COMPLEXITY",
                        f"분석 완료: 점수 {complexity_result.score}",
                        asdict(complexity_result)
                    )

                    # 복잡도 분석기에 결과 표시
                    if self.complexity_analyzer:
                        self.root.after(0, lambda: self.complexity_analyzer.display_complexity_analysis(complexity_result))

                    # 테스트 결과 표시
                    result_text = f"복잡도 점수: {complexity_result.score}/100\n"
                    result_text += f"추론 필요: {'예' if complexity_result.reasoning_required else '아니오'}\n"
                    result_text += f"도메인: {complexity_result.domain}\n"
                    result_text += f"신뢰도: {complexity_result.confidence:.2f}\n"

                    self._update_test_result(result_text)

                else:
                    self.add_debug_entry(DebugLevel.ERROR, "COMPLEXITY", f"분석 실패: {result.error}")

                loop.close()

            except Exception as e:
                self.add_debug_entry(DebugLevel.ERROR, "TEST", f"복잡도 테스트 오류: {e}")

        threading.Thread(target=run_test, daemon=True).start()

    def _test_reasoning(self):
        """추론 체인 테스트"""
        def run_test():
            try:
                test_text = self.test_input.get(1.0, tk.END).strip()
                if not test_text:
                    return

                self.add_debug_entry(DebugLevel.INFO, "TEST", f"추론 체인 시작: {test_text[:30]}...")

                # 비동기 추론 체인
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # 먼저 복잡도 분석
                complexity_result = loop.run_until_complete(
                    self.complexity_detector.detect_complexity(test_text)
                )

                if complexity_result.is_success:
                    complexity_score = complexity_result.value.score

                    # 추론 체인 실행
                    reasoning_result = loop.run_until_complete(
                        self.reasoning_chain.execute_reasoning_chain(test_text, complexity_score)
                    )

                    if reasoning_result.is_success:
                        result = reasoning_result.value
                        self.add_debug_entry(
                            DebugLevel.INFO,
                            "REASONING",
                            f"추론 완료: {len(result.steps)}단계",
                            {
                                'steps_count': len(result.steps),
                                'confidence': result.overall_confidence,
                                'reasoning_type': result.reasoning_type.value
                            }
                        )

                        # 추론 표시기에 결과 표시
                        if self.reasoning_display:
                            self.root.after(0, lambda: self.reasoning_display.show_reasoning_steps(result.steps))

                        # 테스트 결과 표시
                        result_text = f"추론 단계: {len(result.steps)}개\n"
                        result_text += f"추론 유형: {result.reasoning_type.value}\n"
                        result_text += f"전체 신뢰도: {result.overall_confidence:.2f}\n"
                        result_text += "단계별 요약:\n"

                        for i, step in enumerate(result.steps[:5], 1):  # 최대 5단계만 표시
                            result_text += f"  {i}. {step.description[:50]}...\n"

                        self._update_test_result(result_text)

                    else:
                        self.add_debug_entry(DebugLevel.ERROR, "REASONING", f"추론 실패: {reasoning_result.error}")

                loop.close()

            except Exception as e:
                self.add_debug_entry(DebugLevel.ERROR, "TEST", f"추론 테스트 오류: {e}")

        threading.Thread(target=run_test, daemon=True).start()

    def _run_comprehensive_test(self):
        """종합 테스트 실행"""
        def run_test():
            try:
                test_text = self.test_input.get(1.0, tk.END).strip()
                if not test_text:
                    return

                self.add_debug_entry(DebugLevel.INFO, "TEST", "종합 테스트 시작")

                # 복잡도 분석 실행
                self._test_complexity()

                # 잠시 대기 후 추론 테스트
                threading.Timer(2.0, self._test_reasoning).start()

                # 결과 요약
                summary_text = "종합 테스트가 시작되었습니다.\n"
                summary_text += "복잡도 분석과 추론 체인이 순차적으로 실행됩니다.\n"
                summary_text += "각 탭에서 상세 결과를 확인하세요."

                self._update_test_result(summary_text)

            except Exception as e:
                self.add_debug_entry(DebugLevel.ERROR, "TEST", f"종합 테스트 오류: {e}")

        threading.Thread(target=run_test, daemon=True).start()

    def _update_test_result(self, result_text: str):
        """테스트 결과 업데이트"""
        if hasattr(self, 'test_result_display'):
            self.test_result_display.config(state=tk.NORMAL)
            self.test_result_display.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}]\n")
            self.test_result_display.insert(tk.END, result_text)
            self.test_result_display.insert(tk.END, "\n" + "="*50 + "\n")
            self.test_result_display.see(tk.END)
            self.test_result_display.config(state=tk.DISABLED)

    def _clear_debug_log(self):
        """디버그 로그 지우기"""
        self.debug_entries.clear()
        if hasattr(self, 'debug_log_display'):
            self.debug_log_display.config(state=tk.NORMAL)
            self.debug_log_display.delete(1.0, tk.END)
            self.debug_log_display.config(state=tk.DISABLED)

    def _filter_debug_log(self, event=None):
        """디버그 로그 필터링"""
        self._refresh_debug_log()

    def _refresh_debug_log(self):
        """디버그 로그 새로고침"""
        if hasattr(self, 'debug_log_display'):
            self.debug_log_display.config(state=tk.NORMAL)
            self.debug_log_display.delete(1.0, tk.END)
            self.debug_log_display.config(state=tk.DISABLED)

            for entry in self.debug_entries:
                self._update_debug_log_display(entry)

    def _run_test(self):
        """테스트 실행"""
        self._run_comprehensive_test()

    def _open_debug_settings(self):
        """디버그 설정 열기"""
        messagebox.showinfo("디버그 설정", "디버그 설정 기능은 향후 구현 예정입니다.")

    def _export_statistics(self):
        """통계 내보내기"""
        if self.complexity_analyzer:
            stats = self.complexity_analyzer.get_analysis_statistics()
            stats_text = json.dumps(stats, indent=2, ensure_ascii=False)
            messagebox.showinfo("통계", f"복잡도 분석 통계:\n\n{stats_text}")
        else:
            messagebox.showinfo("통계", "표시할 통계가 없습니다.")

    def run(self):
        """디버그 패널 실행 (독립 실행 모드)"""
        if self.standalone and self.root:
            try:
                self.root.mainloop()
            except KeyboardInterrupt:
                print("Debug Panel 종료")


# 편의 함수들
def create_debug_panel(standalone: bool = True) -> DebugPanel:
    """디버그 패널 인스턴스 생성"""
    return DebugPanel(standalone=standalone)


def launch_debug_window():
    """독립 디버그 창 실행"""
    panel = create_debug_panel(standalone=True)
    panel.run()


if __name__ == "__main__":
    # 테스트 실행
    print("PACA v5 Debug Panel 시작...")

    debug_panel = create_debug_panel(standalone=True)
    debug_panel.run()