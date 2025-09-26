"""
Enhanced GUI Application - PACA Python v5
실시간 채팅 및 모니터링 인터페이스

Phase 1-3 시스템과 통합된 데스크톱 GUI 애플리케이션
- 실시간 채팅 인터페이스
- IIS 점수 시각화
- 전술/휴리스틱 관리 패널
- 백업/복원 인터페이스
- 성능 모니터링 대시보드
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import asyncio
import threading
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain
    from ..paca.learning import IISCalculator, AutonomousTrainer, TacticGenerator
    from ..paca.performance import HardwareMonitor, ProfileManager, ProfileType
    from ..paca.core.types.base import Result, create_success, create_failure
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain
    from paca.learning import IISCalculator, AutonomousTrainer, TacticGenerator
    from paca.performance import HardwareMonitor, ProfileManager, ProfileType
    from paca.core.types.base import Result, create_success, create_failure


@dataclass
class ChatMessage:
    """채팅 메시지 데이터"""
    timestamp: datetime
    sender: str  # 'user' or 'assistant'
    content: str
    complexity_score: Optional[int] = None
    reasoning_chain: Optional[List[str]] = None


@dataclass
class LearningStatus:
    """학습 상태 정보"""
    iis_score: int
    trend: str
    tactics_count: int
    heuristics_count: int
    recent_improvements: List[str]


class ChatInterface:
    """실시간 채팅 인터페이스"""

    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self.messages: List[ChatMessage] = []
        self.message_callbacks: List[Callable[[ChatMessage], None]] = []

        # 인지 시스템 초기화
        self.complexity_detector = ComplexityDetector()
        self.reasoning_chain = ReasoningChain()
        self.metacognition = MetacognitionEngine()

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성 요소 초기화"""
        # 채팅 기록 표시 영역
        self.chat_frame = tk.Frame(self.parent_frame)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 채팅 내용 스크롤 텍스트
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Arial', 10),
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # 입력 영역
        self.input_frame = tk.Frame(self.parent_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        # 메시지 입력 필드
        self.message_entry = tk.Entry(
            self.input_frame,
            font=('Arial', 11),
            width=60
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind('<Return>', self._on_send_message)

        # 전송 버튼
        self.send_button = tk.Button(
            self.input_frame,
            text="전송",
            command=self._on_send_message,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.send_button.pack(side=tk.RIGHT)

        # 복잡도 표시 레이블
        self.complexity_label = tk.Label(
            self.parent_frame,
            text="복잡도: 대기 중...",
            font=('Arial', 9),
            fg='#666'
        )
        self.complexity_label.pack(pady=2)

    def add_message_callback(self, callback: Callable[[ChatMessage], None]):
        """메시지 콜백 추가"""
        self.message_callbacks.append(callback)

    def _on_send_message(self, event=None):
        """메시지 전송 처리"""
        message_text = self.message_entry.get().strip()
        if not message_text:
            return

        # 사용자 메시지 추가
        user_message = ChatMessage(
            timestamp=datetime.now(),
            sender='user',
            content=message_text
        )

        self._add_message_to_display(user_message)
        self.message_entry.delete(0, tk.END)

        # 비동기적으로 AI 응답 처리
        threading.Thread(
            target=self._process_ai_response,
            args=(user_message,),
            daemon=True
        ).start()

    def _process_ai_response(self, user_message: ChatMessage):
        """AI 응답 생성 및 처리"""
        try:
            # 복잡도 감지
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            complexity_result = loop.run_until_complete(
                self.complexity_detector.detect_complexity(user_message.content)
            )

            if complexity_result.is_success:
                complexity_score = complexity_result.value.score

                # 복잡도 표시 업데이트
                self.parent_frame.after(
                    0,
                    lambda: self.complexity_label.config(
                        text=f"복잡도: {complexity_score}/100"
                    )
                )

                # 추론 체인 실행 (복잡도 30 이상)
                reasoning_steps = []
                if complexity_score >= 30:
                    reasoning_result = loop.run_until_complete(
                        self.reasoning_chain.execute_reasoning_chain(
                            user_message.content, complexity_score
                        )
                    )

                    if reasoning_result.is_success:
                        reasoning_steps = [
                            step.description for step in reasoning_result.value.steps
                        ]

                # AI 응답 생성 (시뮬레이션)
                response_content = self._generate_response(
                    user_message.content, complexity_score, reasoning_steps
                )

                # AI 메시지 생성
                ai_message = ChatMessage(
                    timestamp=datetime.now(),
                    sender='assistant',
                    content=response_content,
                    complexity_score=complexity_score,
                    reasoning_chain=reasoning_steps
                )

                # UI 업데이트 (메인 스레드에서)
                self.parent_frame.after(
                    0,
                    lambda: self._add_message_to_display(ai_message)
                )

                # 콜백 호출
                for callback in self.message_callbacks:
                    try:
                        callback(ai_message)
                    except Exception as e:
                        logging.warning(f"메시지 콜백 오류: {e}")

            loop.close()

        except Exception as e:
            error_message = ChatMessage(
                timestamp=datetime.now(),
                sender='system',
                content=f"오류가 발생했습니다: {e}"
            )
            self.parent_frame.after(
                0,
                lambda: self._add_message_to_display(error_message)
            )

    def _generate_response(self,
                          user_input: str,
                          complexity_score: int,
                          reasoning_steps: List[str]) -> str:
        """AI 응답 생성 (시뮬레이션)"""
        if complexity_score < 20:
            return f"간단한 질문이네요! '{user_input}'에 대해 답변드리겠습니다."
        elif complexity_score < 50:
            response = f"흥미로운 질문입니다 (복잡도: {complexity_score}).\n"
            if reasoning_steps:
                response += f"추론 과정: {len(reasoning_steps)}단계를 거쳐 분석했습니다."
            return response
        else:
            response = f"매우 복잡한 질문입니다 (복잡도: {complexity_score}).\n"
            if reasoning_steps:
                response += "상세한 추론 과정을 통해 분석한 결과:\n"
                for i, step in enumerate(reasoning_steps[:3], 1):
                    response += f"{i}. {step}\n"
            return response

    def _add_message_to_display(self, message: ChatMessage):
        """채팅 화면에 메시지 추가"""
        self.chat_display.config(state=tk.NORMAL)

        # 타임스탬프와 발신자
        timestamp_str = message.timestamp.strftime("%H:%M:%S")
        sender_prefix = {
            'user': '사용자',
            'assistant': 'PACA',
            'system': '시스템'
        }.get(message.sender, message.sender)

        # 메시지 헤더
        header = f"[{timestamp_str}] {sender_prefix}"
        if message.complexity_score is not None:
            header += f" (복잡도: {message.complexity_score})"
        header += ":\n"

        # 색상 설정
        color_map = {
            'user': '#0066CC',
            'assistant': '#009900',
            'system': '#CC6600'
        }
        color = color_map.get(message.sender, '#000000')

        # 텍스트 삽입
        self.chat_display.insert(tk.END, header, f"header_{message.sender}")
        self.chat_display.insert(tk.END, message.content + "\n\n")

        # 스타일 적용
        self.chat_display.tag_config(f"header_{message.sender}", foreground=color, font=('Arial', 10, 'bold'))

        # 자동 스크롤
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

        # 메시지 히스토리에 추가
        self.messages.append(message)

    def clear_chat(self):
        """채팅 기록 지우기"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.messages.clear()


class MonitoringPanel:
    """실시간 시스템 모니터링 패널"""

    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self.hardware_monitor = HardwareMonitor(monitoring_interval=2.0)
        self.profile_manager = ProfileManager()

        # 모니터링 상태
        self.monitoring_active = False
        self.update_job = None

        self._setup_ui()
        self._setup_monitoring()

    def _setup_ui(self):
        """모니터링 UI 구성"""
        # 제목
        title_label = tk.Label(
            self.parent_frame,
            text="시스템 성능 모니터링",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 시스템 상태 프레임
        status_frame = tk.LabelFrame(self.parent_frame, text="시스템 상태", font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        # CPU 사용률
        self.cpu_label = tk.Label(status_frame, text="CPU: 0.0%", font=('Arial', 10))
        self.cpu_label.pack(anchor=tk.W, padx=10, pady=2)

        self.cpu_bar = ttk.Progressbar(status_frame, length=300, mode='determinate')
        self.cpu_bar.pack(padx=10, pady=2)

        # 메모리 사용률
        self.memory_label = tk.Label(status_frame, text="Memory: 0.0%", font=('Arial', 10))
        self.memory_label.pack(anchor=tk.W, padx=10, pady=2)

        self.memory_bar = ttk.Progressbar(status_frame, length=300, mode='determinate')
        self.memory_bar.pack(padx=10, pady=2)

        # 건강도 점수
        self.health_label = tk.Label(status_frame, text="건강도: 100.0", font=('Arial', 10))
        self.health_label.pack(anchor=tk.W, padx=10, pady=2)

        # 성능 프로파일 프레임
        profile_frame = tk.LabelFrame(self.parent_frame, text="성능 프로파일", font=('Arial', 11, 'bold'))
        profile_frame.pack(fill=tk.X, padx=10, pady=5)

        # 현재 프로파일
        self.current_profile_label = tk.Label(
            profile_frame,
            text="현재 프로파일: mid-range",
            font=('Arial', 10)
        )
        self.current_profile_label.pack(anchor=tk.W, padx=10, pady=2)

        # 프로파일 전환 버튼들
        button_frame = tk.Frame(profile_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        profiles = [
            ("Conservative", ProfileType.CONSERVATIVE),
            ("Low-End", ProfileType.LOW_END),
            ("Mid-Range", ProfileType.MID_RANGE),
            ("High-End", ProfileType.HIGH_END)
        ]

        for text, profile_type in profiles:
            btn = tk.Button(
                button_frame,
                text=text,
                command=lambda pt=profile_type: self._switch_profile(pt),
                width=12
            )
            btn.pack(side=tk.LEFT, padx=2)

        # 알림 영역
        alert_frame = tk.LabelFrame(self.parent_frame, text="알림", font=('Arial', 11, 'bold'))
        alert_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.alert_display = scrolledtext.ScrolledText(
            alert_frame,
            height=6,
            font=('Arial', 9),
            state=tk.DISABLED
        )
        self.alert_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 제어 버튼
        control_frame = tk.Frame(self.parent_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.monitor_button = tk.Button(
            control_frame,
            text="모니터링 시작",
            command=self._toggle_monitoring,
            bg='#4CAF50',
            fg='white'
        )
        self.monitor_button.pack(side=tk.LEFT, padx=5)

        self.refresh_button = tk.Button(
            control_frame,
            text="즉시 새로고침",
            command=self._refresh_status
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)

    def _setup_monitoring(self):
        """모니터링 콜백 설정"""
        def status_callback(status):
            # UI 업데이트를 메인 스레드에서 실행
            self.parent_frame.after(0, lambda: self._update_status_display(status))

        self.hardware_monitor.add_callback(status_callback)

    def _update_status_display(self, status):
        """상태 디스플레이 업데이트"""
        try:
            # CPU 업데이트
            cpu_percent = status.resource_usage.cpu_percent
            self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")
            self.cpu_bar['value'] = cpu_percent

            # 메모리 업데이트
            memory_percent = status.resource_usage.memory_percent
            self.memory_label.config(text=f"Memory: {memory_percent:.1f}%")
            self.memory_bar['value'] = memory_percent

            # 건강도 업데이트
            health_score = status.overall_health_score
            self.health_label.config(text=f"건강도: {health_score:.1f}")

            # 프로파일 업데이트
            recommended = status.recommended_profile
            current = self.profile_manager.current_profile.name
            self.current_profile_label.config(
                text=f"현재: {current} | 추천: {recommended}"
            )

            # 알림 표시
            if status.alerts:
                self._add_alert(f"[{datetime.now().strftime('%H:%M:%S')}] {len(status.alerts)}개 알림 발생")
                for alert in status.alerts:
                    self._add_alert(f"  - {alert.level.value}: {alert.message}")

        except Exception as e:
            self._add_alert(f"상태 업데이트 오류: {e}")

    def _add_alert(self, message: str):
        """알림 메시지 추가"""
        self.alert_display.config(state=tk.NORMAL)
        self.alert_display.insert(tk.END, message + "\n")
        self.alert_display.see(tk.END)
        self.alert_display.config(state=tk.DISABLED)

    def _toggle_monitoring(self):
        """모니터링 시작/중지"""
        if self.monitoring_active:
            # 모니터링 중지
            asyncio.run(self.hardware_monitor.stop_monitoring())
            self.monitoring_active = False
            self.monitor_button.config(text="모니터링 시작", bg='#4CAF50')
            self._add_alert("모니터링이 중지되었습니다.")
        else:
            # 모니터링 시작
            def start_monitoring():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.hardware_monitor.start_monitoring())

            threading.Thread(target=start_monitoring, daemon=True).start()
            self.monitoring_active = True
            self.monitor_button.config(text="모니터링 중지", bg='#f44336')
            self._add_alert("실시간 모니터링이 시작되었습니다.")

    def _refresh_status(self):
        """즉시 상태 새로고침"""
        def refresh():
            try:
                result = self.hardware_monitor.get_system_status()
                if result.is_success:
                    self.parent_frame.after(0, lambda: self._update_status_display(result.value))
                else:
                    self.parent_frame.after(0, lambda: self._add_alert(f"상태 조회 실패: {result.error}"))
            except Exception as e:
                self.parent_frame.after(0, lambda: self._add_alert(f"새로고침 오류: {e}"))

        threading.Thread(target=refresh, daemon=True).start()

    def _switch_profile(self, profile_type: ProfileType):
        """프로파일 전환"""
        result = self.profile_manager.switch_profile(profile_type, "사용자 수동 전환")
        if result.is_success:
            profile = result.value
            self._add_alert(f"프로파일 전환: {profile.name}")
            self.current_profile_label.config(text=f"현재: {profile.name}")
        else:
            self._add_alert(f"프로파일 전환 실패: {result.error}")


class BackupManager:
    """학습 데이터 백업/복원 관리자"""

    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self._setup_ui()

    def _setup_ui(self):
        """백업 관리 UI 구성"""
        # 제목
        title_label = tk.Label(
            self.parent_frame,
            text="학습 데이터 백업/복원",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # 백업 생성 프레임
        backup_frame = tk.LabelFrame(self.parent_frame, text="백업 생성", font=('Arial', 11, 'bold'))
        backup_frame.pack(fill=tk.X, padx=10, pady=5)

        backup_button = tk.Button(
            backup_frame,
            text="현재 상태 백업",
            command=self._create_backup,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        backup_button.pack(pady=10)

        # 백업 목록 프레임
        list_frame = tk.LabelFrame(self.parent_frame, text="백업 목록", font=('Arial', 11, 'bold'))
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 백업 목록 트리뷰
        columns = ('날짜', '시간', '크기', '설명')
        self.backup_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)

        for col in columns:
            self.backup_tree.heading(col, text=col)
            self.backup_tree.column(col, width=120)

        self.backup_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 복원 버튼
        restore_frame = tk.Frame(self.parent_frame)
        restore_frame.pack(fill=tk.X, padx=10, pady=5)

        restore_button = tk.Button(
            restore_frame,
            text="선택된 백업 복원",
            command=self._restore_backup,
            bg='#FF9800',
            fg='white'
        )
        restore_button.pack(side=tk.LEFT, padx=5)

        delete_button = tk.Button(
            restore_frame,
            text="백업 삭제",
            command=self._delete_backup,
            bg='#f44336',
            fg='white'
        )
        delete_button.pack(side=tk.LEFT, padx=5)

        # 초기 백업 목록 로드
        self._load_backup_list()

    def _create_backup(self):
        """백업 생성"""
        try:
            timestamp = datetime.now()
            backup_name = f"PACA_BACKUP_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            # 실제 백업 로직은 여기에 구현
            # 현재는 시뮬레이션
            backup_info = {
                'name': backup_name,
                'timestamp': timestamp.isoformat(),
                'size': '2.5MB',  # 시뮬레이션
                'description': '수동 백업'
            }

            # 백업 목록에 추가
            self.backup_tree.insert('', 0, values=(
                timestamp.strftime('%Y-%m-%d'),
                timestamp.strftime('%H:%M:%S'),
                backup_info['size'],
                backup_info['description']
            ))

            messagebox.showinfo("백업 완료", f"백업이 성공적으로 생성되었습니다.\n백업명: {backup_name}")

        except Exception as e:
            messagebox.showerror("백업 실패", f"백업 생성 중 오류가 발생했습니다:\n{e}")

    def _restore_backup(self):
        """선택된 백업 복원"""
        selection = self.backup_tree.selection()
        if not selection:
            messagebox.showwarning("선택 오류", "복원할 백업을 선택해주세요.")
            return

        item = self.backup_tree.item(selection[0])
        backup_date = item['values'][0]
        backup_time = item['values'][1]

        # 확인 대화상자
        if messagebox.askyesno("백업 복원", f"{backup_date} {backup_time} 백업으로 복원하시겠습니까?\n\n현재 학습 데이터가 손실될 수 있습니다."):
            try:
                # 실제 복원 로직은 여기에 구현
                messagebox.showinfo("복원 완료", "백업이 성공적으로 복원되었습니다.")
            except Exception as e:
                messagebox.showerror("복원 실패", f"백업 복원 중 오류가 발생했습니다:\n{e}")

    def _delete_backup(self):
        """선택된 백업 삭제"""
        selection = self.backup_tree.selection()
        if not selection:
            messagebox.showwarning("선택 오류", "삭제할 백업을 선택해주세요.")
            return

        item = self.backup_tree.item(selection[0])
        backup_date = item['values'][0]
        backup_time = item['values'][1]

        if messagebox.askyesno("백업 삭제", f"{backup_date} {backup_time} 백업을 삭제하시겠습니까?"):
            try:
                # 실제 삭제 로직은 여기에 구현
                self.backup_tree.delete(selection[0])
                messagebox.showinfo("삭제 완료", "백업이 삭제되었습니다.")
            except Exception as e:
                messagebox.showerror("삭제 실패", f"백업 삭제 중 오류가 발생했습니다:\n{e}")

    def _load_backup_list(self):
        """백업 목록 로드"""
        # 시뮬레이션 데이터
        sample_backups = [
            ('2024-09-21', '14:30:15', '2.3MB', '자동 백업'),
            ('2024-09-21', '10:15:30', '2.1MB', 'Phase 2 완료 후'),
            ('2024-09-20', '18:45:20', '1.9MB', '수동 백업'),
        ]

        for backup in sample_backups:
            self.backup_tree.insert('', tk.END, values=backup)


class EnhancedGUI:
    """
    PACA v5 Enhanced GUI 메인 애플리케이션

    실시간 채팅, 시스템 모니터링, 학습 상태 관리를 통합한
    데스크톱 GUI 애플리케이션
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PACA v5 Enhanced GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # 학습 시스템 초기화
        self.iis_calculator = None
        self.tactic_generator = None
        self.learning_status: Optional[LearningStatus] = None

        # UI 구성 요소
        self.chat_interface: Optional[ChatInterface] = None
        self.monitoring_panel: Optional[MonitoringPanel] = None
        self.backup_manager: Optional[BackupManager] = None

        self._setup_ui()
        self._setup_learning_systems()

    def _setup_ui(self):
        """메인 UI 구성"""
        # 메뉴 바
        self._create_menu()

        # 메인 프레임
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 탭 노트북
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 채팅 탭
        chat_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(chat_frame, text="💬 채팅")
        self.chat_interface = ChatInterface(chat_frame)

        # 모니터링 탭
        monitor_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(monitor_frame, text="📊 모니터링")
        self.monitoring_panel = MonitoringPanel(monitor_frame)

        # 백업 탭
        backup_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(backup_frame, text="💾 백업")
        self.backup_manager = BackupManager(backup_frame)

        # 학습 상태 탭
        learning_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(learning_frame, text="🎓 학습 상태")
        self._setup_learning_tab(learning_frame)

        # 상태 바
        self._create_status_bar()

    def _create_menu(self):
        """메뉴 바 생성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="설정", command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)

        # 도구 메뉴
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="디버그 모드", command=self._open_debug_mode)
        tools_menu.add_command(label="성능 분석", command=self._open_performance_analysis)

        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용법", command=self._show_help)
        help_menu.add_command(label="정보", command=self._show_about)

    def _setup_learning_tab(self, parent_frame: tk.Frame):
        """학습 상태 탭 구성"""
        # 제목
        title_label = tk.Label(
            parent_frame,
            text="PACA 학습 상태",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=10)

        # IIS 점수 프레임
        iis_frame = tk.LabelFrame(parent_frame, text="IIS (지능적 지위성) 점수", font=('Arial', 12, 'bold'))
        iis_frame.pack(fill=tk.X, padx=20, pady=10)

        self.iis_score_label = tk.Label(
            iis_frame,
            text="IIS 점수: 75/100 (B등급)",
            font=('Arial', 14, 'bold'),
            fg='#2196F3'
        )
        self.iis_score_label.pack(pady=10)

        self.iis_trend_label = tk.Label(
            iis_frame,
            text="추세: 상승 중 ↗",
            font=('Arial', 11),
            fg='#4CAF50'
        )
        self.iis_trend_label.pack()

        # 전술/휴리스틱 프레임
        tactics_frame = tk.LabelFrame(parent_frame, text="학습된 전술 및 휴리스틱", font=('Arial', 12, 'bold'))
        tactics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 통계 표시
        stats_frame = tk.Frame(tactics_frame)
        stats_frame.pack(fill=tk.X, pady=5)

        self.tactics_count_label = tk.Label(
            stats_frame,
            text="전술: 23개",
            font=('Arial', 11)
        )
        self.tactics_count_label.pack(side=tk.LEFT, padx=20)

        self.heuristics_count_label = tk.Label(
            stats_frame,
            text="휴리스틱: 15개",
            font=('Arial', 11)
        )
        self.heuristics_count_label.pack(side=tk.LEFT, padx=20)

        # 최근 개선 사항
        improvements_label = tk.Label(
            tactics_frame,
            text="최근 개선 사항:",
            font=('Arial', 11, 'bold')
        )
        improvements_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        self.improvements_display = scrolledtext.ScrolledText(
            tactics_frame,
            height=8,
            font=('Arial', 10),
            state=tk.DISABLED
        )
        self.improvements_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 훈련 버튼
        training_button = tk.Button(
            parent_frame,
            text="자율 훈련 시작",
            command=self._start_autonomous_training,
            bg='#FF9800',
            fg='white',
            font=('Arial', 12, 'bold')
        )
        training_button.pack(pady=10)

    def _create_status_bar(self):
        """상태 바 생성"""
        self.status_bar = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            self.status_bar,
            text="PACA v5 Enhanced GUI 준비됨",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 현재 시간 표시
        self.time_label = tk.Label(
            self.status_bar,
            text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            anchor=tk.E
        )
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # 시간 업데이트
        self._update_time()

    def _update_time(self):
        """시간 업데이트"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self._update_time)

    def _setup_learning_systems(self):
        """학습 시스템 초기화"""
        try:
            self.iis_calculator = IISCalculator()
            self.tactic_generator = TacticGenerator()

            # 초기 학습 상태 로드
            self._update_learning_status()

            # 채팅 메시지 콜백 등록
            if self.chat_interface:
                self.chat_interface.add_message_callback(self._on_chat_message)

        except Exception as e:
            logging.error(f"학습 시스템 초기화 실패: {e}")

    def _update_learning_status(self):
        """학습 상태 업데이트"""
        try:
            # 시뮬레이션 데이터
            self.learning_status = LearningStatus(
                iis_score=75,
                trend="improving",
                tactics_count=23,
                heuristics_count=15,
                recent_improvements=[
                    "복잡도 감지 정확도 5% 향상",
                    "추론 체인 효율성 개선",
                    "새로운 분석적 전술 학습"
                ]
            )

            # UI 업데이트
            if hasattr(self, 'iis_score_label'):
                self.iis_score_label.config(text=f"IIS 점수: {self.learning_status.iis_score}/100 (B등급)")

            if hasattr(self, 'tactics_count_label'):
                self.tactics_count_label.config(text=f"전술: {self.learning_status.tactics_count}개")

            if hasattr(self, 'heuristics_count_label'):
                self.heuristics_count_label.config(text=f"휴리스틱: {self.learning_status.heuristics_count}개")

            # 개선 사항 업데이트
            if hasattr(self, 'improvements_display'):
                self.improvements_display.config(state=tk.NORMAL)
                self.improvements_display.delete(1.0, tk.END)
                for improvement in self.learning_status.recent_improvements:
                    self.improvements_display.insert(tk.END, f"• {improvement}\n")
                self.improvements_display.config(state=tk.DISABLED)

        except Exception as e:
            logging.error(f"학습 상태 업데이트 실패: {e}")

    def _on_chat_message(self, message: ChatMessage):
        """채팅 메시지 처리 콜백"""
        # 학습 데이터로 활용할 수 있는 로직
        if message.sender == 'user' and message.complexity_score:
            self.status_label.config(text=f"복잡도 {message.complexity_score} 메시지 처리됨")

    def _start_autonomous_training(self):
        """자율 훈련 시작"""
        def training_thread():
            try:
                self.status_label.config(text="자율 훈련 진행 중...")
                # 실제 훈련 로직은 여기에 구현
                self.root.after(3000, lambda: self.status_label.config(text="자율 훈련 완료"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("훈련 오류", f"자율 훈련 중 오류 발생: {e}"))

        threading.Thread(target=training_thread, daemon=True).start()

    def _open_settings(self):
        """설정 창 열기"""
        messagebox.showinfo("설정", "설정 기능은 향후 구현 예정입니다.")

    def _open_debug_mode(self):
        """디버그 모드 열기"""
        messagebox.showinfo("디버그 모드", "디버그 패널은 별도 창에서 실행됩니다.")

    def _open_performance_analysis(self):
        """성능 분석 열기"""
        messagebox.showinfo("성능 분석", "성능 분석 도구는 모니터링 탭에서 확인할 수 있습니다.")

    def _show_help(self):
        """도움말 표시"""
        help_text = """
PACA v5 Enhanced GUI 사용법

1. 채팅 탭: AI와 실시간 대화 및 복잡도 분석
2. 모니터링 탭: 시스템 성능 실시간 모니터링
3. 백업 탭: 학습 데이터 백업 및 복원
4. 학습 상태 탭: IIS 점수 및 학습 진행 상황

각 탭에서 제공되는 기능을 활용하여
PACA 시스템을 효율적으로 관리하세요.
        """
        messagebox.showinfo("도움말", help_text.strip())

    def _show_about(self):
        """정보 표시"""
        about_text = """
PACA v5 Enhanced GUI
버전: 5.0.0

PACA (Python Adaptive Cognitive Architecture)
Python 기반 적응형 인지 아키텍처

개발: PACA Development Team
날짜: 2024-09-21
        """
        messagebox.showinfo("정보", about_text.strip())

    def update_learning_status(self, iis_score: int, tactics: list):
        """외부에서 학습 상태 업데이트"""
        if self.learning_status:
            self.learning_status.iis_score = iis_score
            self.learning_status.tactics_count = len(tactics)
            self._update_learning_status()

    def run(self):
        """GUI 실행"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("GUI 종료")
        finally:
            # 정리 작업
            if self.monitoring_panel and self.monitoring_panel.monitoring_active:
                asyncio.run(self.monitoring_panel.hardware_monitor.stop_monitoring())


# 편의 함수들
def create_enhanced_gui() -> EnhancedGUI:
    """Enhanced GUI 인스턴스 생성"""
    return EnhancedGUI()


if __name__ == "__main__":
    # 테스트 실행
    print("PACA v5 Enhanced GUI 시작...")

    app = EnhancedGUI()
    app.run()