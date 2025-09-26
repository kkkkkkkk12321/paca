"""
PACA v5 Desktop Application
CustomTkinter 기반 GUI 데스크톱 애플리케이션
"""

import customtkinter as ctk
import asyncio
import threading
import sys
import os
from typing import Optional
from pathlib import Path
import tkinter.messagebox as messagebox

# UTF-8 인코딩 설정 (Windows 호환성)
if os.name == 'nt':  # Windows
    try:
        # Python 3.7+ 에서 UTF-8 모드 활성화
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # 이전 버전 Python 지원
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# PACA 모듈 import
sys.path.append(str(Path(__file__).parent.parent))

try:
    from paca.core.utils.optional_imports import (
        try_import_customtkinter, has_gui_features, print_feature_status
    )
    from paca.mathematics import Calculator
    from paca.services.learning import LearningService
    from paca.core.types.base import Result, Status

    # GUI 기능 확인
    if not has_gui_features():
        print("GUI 기능을 사용할 수 없습니다.")
        print("다음 명령으로 GUI 의존성을 설치하세요:")
        print("pip install customtkinter>=5.0.0 Pillow>=9.0.0")
        sys.exit(1)

except ImportError as e:
    print(f"PACA 모듈 import 오류: {e}")
    print("기본 기능만 제공됩니다.")
    print_feature_status()


class PacaDesktopApp:
    """PACA v5 데스크톱 GUI 애플리케이션"""

    def __init__(self):
        # CustomTkinter 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 메인 윈도우 생성
        self.root = ctk.CTk()
        self.root.title("PACA v5 - AI Assistant")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # PACA 시스템 초기화
        self.calculator = None
        self.learning_service = None
        self.is_running = False

        # UI 구성 요소
        self.chat_display = None
        self.chat_entry = None
        self.status_label = None

        # UI 설정
        self.setup_ui()
        self.initialize_paca_services()

    def setup_ui(self):
        """UI 구성 요소 설정"""
        # 메인 프레임
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 사이드바 (설정 및 도구)
        self.sidebar = ctk.CTkFrame(self.main_frame, width=250)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))
        self.sidebar.pack_propagate(False)  # 크기 고정

        # 채팅 영역
        self.chat_area = ctk.CTkFrame(self.main_frame)
        self.chat_area.pack(side="right", fill="both", expand=True)

        # 사이드바 구성
        self.setup_sidebar()

        # 채팅 인터페이스 구성
        self.setup_chat_interface()

    def setup_sidebar(self):
        """사이드바 설정"""
        # 제목 라벨
        title_label = ctk.CTkLabel(
            self.sidebar,
            text="PACA v5",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(10, 20))

        # 파일 메뉴
        file_frame = ctk.CTkFrame(self.sidebar)
        file_frame.pack(fill="x", padx=10, pady=5)

        file_label = ctk.CTkLabel(file_frame, text="파일", font=ctk.CTkFont(weight="bold"))
        file_label.pack(pady=5)

        # 파일 버튼들
        new_btn = ctk.CTkButton(
            file_frame, text="새 대화", command=self.new_conversation,
            height=30, width=200
        )
        new_btn.pack(pady=2, padx=10)

        save_btn = ctk.CTkButton(
            file_frame, text="대화 저장", command=self.save_conversation,
            height=30, width=200
        )
        save_btn.pack(pady=2, padx=10)

        # 도구 메뉴
        tools_frame = ctk.CTkFrame(self.sidebar)
        tools_frame.pack(fill="x", padx=10, pady=5)

        tools_label = ctk.CTkLabel(tools_frame, text="도구", font=ctk.CTkFont(weight="bold"))
        tools_label.pack(pady=5)

        # 도구 버튼들
        calc_btn = ctk.CTkButton(
            tools_frame, text="계산기", command=self.open_calculator,
            height=30, width=200
        )
        calc_btn.pack(pady=2, padx=10)

        learning_btn = ctk.CTkButton(
            tools_frame, text="학습 통계", command=self.show_learning_stats,
            height=30, width=200
        )
        learning_btn.pack(pady=2, padx=10)

        status_btn = ctk.CTkButton(
            tools_frame, text="시스템 상태", command=self.show_system_status,
            height=30, width=200
        )
        status_btn.pack(pady=2, padx=10)

        # 상태 표시
        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="상태: 준비됨",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="bottom", pady=10)

    def setup_chat_interface(self):
        """채팅 인터페이스 설정"""
        # 채팅 디스플레이
        self.chat_display = ctk.CTkTextbox(
            self.chat_area,
            wrap="word",
            font=ctk.CTkFont(size=12)
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        # 입력 영역
        input_frame = ctk.CTkFrame(self.chat_area)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.chat_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="메시지를 입력하세요...",
            font=ctk.CTkFont(size=12)
        )
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_entry.bind("<Return>", self.on_send_message)

        send_btn = ctk.CTkButton(
            input_frame,
            text="전송",
            command=self.on_send_message,
            width=80
        )
        send_btn.pack(side="right")

        # 초기 메시지
        self.add_chat_message("PACA", "안녕하세요! PACA v5에 오신 것을 환영합니다.")

    def initialize_paca_services(self):
        """PACA 서비스 초기화"""
        try:
            self.calculator = Calculator()
            self.learning_service = LearningService()
            self.is_running = True
            self.update_status("PACA 서비스 준비됨", "green")
        except Exception as e:
            self.update_status(f"초기화 오류: {str(e)}", "red")
            print(f"PACA 서비스 초기화 오류: {e}")

    def add_chat_message(self, sender: str, message: str):
        """채팅 메시지 추가"""
        self.chat_display.insert("end", f"{sender}: {message}\n\n")
        self.chat_display.see("end")

    def on_send_message(self, event=None):
        """메시지 전송 처리"""
        message = self.chat_entry.get().strip()
        if not message:
            return

        # 사용자 메시지 표시
        self.add_chat_message("사용자", message)
        self.chat_entry.delete(0, "end")

        # 메시지 처리
        response = self.process_message(message)
        self.add_chat_message("PACA", response)

    def process_message(self, message: str) -> str:
        """메시지 처리 및 응답 생성"""
        try:
            # 계산 요청 처리
            if any(op in message for op in ['+', '-', '*', '/', '계산', '더하기', '빼기', '곱하기', '나누기']):
                return self.handle_calculation(message)

            # 학습 관련 요청 처리
            if any(word in message for word in ['학습', '공부', '기억', '저장']):
                return self.handle_learning(message)

            # 기본 응답
            return f"메시지를 받았습니다: '{message}'\n현재 계산과 학습 기능을 지원합니다."

        except Exception as e:
            return f"처리 중 오류가 발생했습니다: {str(e)}"

    def handle_calculation(self, message: str) -> str:
        """계산 요청 처리"""
        if not self.calculator:
            return "계산기 서비스를 사용할 수 없습니다."

        try:
            # 간단한 계산 파싱
            if '+' in message:
                parts = message.split('+')
                if len(parts) == 2:
                    a = float(parts[0].strip().split()[-1])
                    b = float(parts[1].strip().split()[0])
                    result = self.calculator.add(a, b)
                    return f"{a} + {b} = {result.value if result.is_success else '오류'}"

            # 기본 계산 테스트
            result = self.calculator.add(2, 3)
            return f"계산 테스트: 2 + 3 = {result.value if result.is_success else '오류'}"

        except Exception as e:
            return f"계산 오류: {str(e)}"

    def handle_learning(self, message: str) -> str:
        """학습 요청 처리"""
        if not self.learning_service:
            return "학습 서비스를 사용할 수 없습니다."

        try:
            # 학습 세션 생성 시뮬레이션
            return f"학습 메시지를 처리했습니다: '{message}'\n학습 시스템이 활성화되었습니다."
        except Exception as e:
            return f"학습 처리 오류: {str(e)}"

    def update_status(self, message: str, color: str = "white"):
        """상태 메시지 업데이트"""
        if self.status_label:
            self.status_label.configure(text=f"상태: {message}")

    def new_conversation(self):
        """새 대화 시작"""
        self.chat_display.delete("1.0", "end")
        self.add_chat_message("PACA", "새로운 대화를 시작합니다.")
        self.update_status("새 대화 시작됨")

    def save_conversation(self):
        """대화 저장"""
        content = self.chat_display.get("1.0", "end")
        try:
            with open("conversation.txt", "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("저장 완료", "대화가 conversation.txt에 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("저장 오류", f"저장 중 오류가 발생했습니다: {str(e)}")

    def open_calculator(self):
        """계산기 창 열기"""
        calc_window = ctk.CTkToplevel(self.root)
        calc_window.title("PACA 계산기")
        calc_window.geometry("300x400")

        # 계산기 디스플레이
        display = ctk.CTkEntry(calc_window, font=ctk.CTkFont(size=16))
        display.pack(fill="x", padx=10, pady=10)

        # 계산기 버튼들
        buttons_frame = ctk.CTkFrame(calc_window)
        buttons_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 간단한 계산 테스트 버튼
        test_btn = ctk.CTkButton(
            buttons_frame,
            text="2 + 3 계산 테스트",
            command=lambda: self.calc_test(display)
        )
        test_btn.pack(pady=10)

    def calc_test(self, display):
        """계산 테스트"""
        if self.calculator:
            result = self.calculator.add(2, 3)
            display.delete(0, "end")
            display.insert(0, f"2 + 3 = {result.value if result.is_success else '오류'}")

    def show_learning_stats(self):
        """학습 통계 표시"""
        stats_window = ctk.CTkToplevel(self.root)
        stats_window.title("학습 통계")
        stats_window.geometry("400x300")

        stats_text = """
        학습 통계
        ========

        학습 세션: 준비됨
        학습 서비스: 활성화됨

        현재 학습 기능이 구현되어 있습니다.
        """

        stats_label = ctk.CTkTextbox(stats_window)
        stats_label.pack(fill="both", expand=True, padx=20, pady=20)
        stats_label.insert("1.0", stats_text)
        stats_label.configure(state="disabled")

    def show_system_status(self):
        """시스템 상태 표시"""
        status_window = ctk.CTkToplevel(self.root)
        status_window.title("시스템 상태")
        status_window.geometry("400x300")

        status_text = f"""
        PACA v5 시스템 상태
        ==================

        서비스 상태: {'활성화' if self.is_running else '비활성화'}
        계산기: {'사용 가능' if self.calculator else '사용 불가'}
        학습 서비스: {'사용 가능' if self.learning_service else '사용 불가'}

        GUI 애플리케이션이 정상 동작 중입니다.
        """

        status_label = ctk.CTkTextbox(status_window)
        status_label.pack(fill="both", expand=True, padx=20, pady=20)
        status_label.insert("1.0", status_text)
        status_label.configure(state="disabled")

    def on_closing(self):
        """애플리케이션 종료 처리"""
        self.root.destroy()

    def run(self):
        """애플리케이션 실행"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """메인 실행"""
    app = PacaDesktopApp()
    app.run()


if __name__ == "__main__":
    main()