"""
설정 패널 UI 컴포넌트
PACA v5 Desktop Application
"""

import customtkinter as ctk
from typing import Callable, Dict, Any


class SettingsPanel:
    """설정 패널 UI"""

    def __init__(self, parent_frame: ctk.CTkFrame, change_callback: Callable):
        self.parent = parent_frame
        self.change_callback = change_callback

        # 설정 값들
        self.settings = {
            "appearance_mode": "dark",
            "color_theme": "blue",
            "enable_learning": True,
            "enable_korean_nlp": True,
            "response_timeout": 5.0,
            "quality_threshold": 0.7,
            "auto_save": True,
            "notification_sound": True
        }

        self.setup_ui()

    def setup_ui(self):
        """설정 UI 구성"""
        # 설정 섹션 프레임
        self.settings_frame = ctk.CTkFrame(self.parent)
        self.settings_frame.pack(fill="x", padx=10, pady=10)

        # 설정 제목
        settings_label = ctk.CTkLabel(
            self.settings_frame,
            text="설정",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        settings_label.pack(pady=10)

        # 외관 설정
        self.setup_appearance_section()

        # AI 설정
        self.setup_ai_section()

        # 일반 설정
        self.setup_general_section()

    def setup_appearance_section(self):
        """외관 설정 섹션"""
        # 외관 섹션 프레임
        appearance_frame = ctk.CTkFrame(self.settings_frame)
        appearance_frame.pack(fill="x", padx=10, pady=5)

        # 섹션 제목
        appearance_title = ctk.CTkLabel(
            appearance_frame,
            text="외관",
            font=ctk.CTkFont(weight="bold")
        )
        appearance_title.pack(pady=(10, 5))

        # 다크/라이트 모드
        mode_frame = ctk.CTkFrame(appearance_frame)
        mode_frame.pack(fill="x", padx=10, pady=2)

        mode_label = ctk.CTkLabel(mode_frame, text="테마 모드:")
        mode_label.pack(side="left", padx=10, pady=5)

        self.mode_var = ctk.StringVar(value=self.settings["appearance_mode"])
        mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            values=["dark", "light", "system"],
            variable=self.mode_var,
            command=self.on_mode_change
        )
        mode_menu.pack(side="right", padx=10, pady=5)

        # 색상 테마
        theme_frame = ctk.CTkFrame(appearance_frame)
        theme_frame.pack(fill="x", padx=10, pady=2)

        theme_label = ctk.CTkLabel(theme_frame, text="색상 테마:")
        theme_label.pack(side="left", padx=10, pady=5)

        self.theme_var = ctk.StringVar(value=self.settings["color_theme"])
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["blue", "green", "dark-blue"],
            variable=self.theme_var,
            command=self.on_theme_change
        )
        theme_menu.pack(side="right", padx=10, pady=5)

        # 설정 적용 버튼
        apply_button = ctk.CTkButton(
            appearance_frame,
            text="외관 설정 적용",
            command=self.apply_appearance_settings,
            height=30
        )
        apply_button.pack(pady=5)

    def setup_ai_section(self):
        """AI 설정 섹션"""
        # AI 섹션 프레임
        ai_frame = ctk.CTkFrame(self.settings_frame)
        ai_frame.pack(fill="x", padx=10, pady=5)

        # 섹션 제목
        ai_title = ctk.CTkLabel(
            ai_frame,
            text="AI 설정",
            font=ctk.CTkFont(weight="bold")
        )
        ai_title.pack(pady=(10, 5))

        # 학습 활성화
        self.learning_var = ctk.BooleanVar(value=self.settings["enable_learning"])
        learning_checkbox = ctk.CTkCheckBox(
            ai_frame,
            text="학습 기능 활성화",
            variable=self.learning_var,
            command=self.on_learning_change
        )
        learning_checkbox.pack(pady=2)

        # 한국어 NLP 활성화
        self.korean_nlp_var = ctk.BooleanVar(value=self.settings["enable_korean_nlp"])
        korean_checkbox = ctk.CTkCheckBox(
            ai_frame,
            text="한국어 NLP 활성화",
            variable=self.korean_nlp_var,
            command=self.on_korean_nlp_change
        )
        korean_checkbox.pack(pady=2)

        # 응답 시간 제한
        timeout_frame = ctk.CTkFrame(ai_frame)
        timeout_frame.pack(fill="x", padx=10, pady=5)

        timeout_label = ctk.CTkLabel(timeout_frame, text="응답 시간 제한 (초):")
        timeout_label.pack(side="left", padx=10, pady=5)

        self.timeout_var = ctk.DoubleVar(value=self.settings["response_timeout"])
        timeout_slider = ctk.CTkSlider(
            timeout_frame,
            from_=1.0,
            to=10.0,
            variable=self.timeout_var,
            command=self.on_timeout_change
        )
        timeout_slider.pack(side="right", padx=10, pady=5, fill="x", expand=True)

        self.timeout_value_label = ctk.CTkLabel(
            timeout_frame,
            text=f"{self.settings['response_timeout']:.1f}s"
        )
        self.timeout_value_label.pack(side="right", padx=5, pady=5)

        # 품질 임계값
        quality_frame = ctk.CTkFrame(ai_frame)
        quality_frame.pack(fill="x", padx=10, pady=5)

        quality_label = ctk.CTkLabel(quality_frame, text="품질 임계값:")
        quality_label.pack(side="left", padx=10, pady=5)

        self.quality_var = ctk.DoubleVar(value=self.settings["quality_threshold"])
        quality_slider = ctk.CTkSlider(
            quality_frame,
            from_=0.1,
            to=1.0,
            variable=self.quality_var,
            command=self.on_quality_change
        )
        quality_slider.pack(side="right", padx=10, pady=5, fill="x", expand=True)

        self.quality_value_label = ctk.CTkLabel(
            quality_frame,
            text=f"{self.settings['quality_threshold']:.1f}"
        )
        self.quality_value_label.pack(side="right", padx=5, pady=5)

    def setup_general_section(self):
        """일반 설정 섹션"""
        # 일반 섹션 프레임
        general_frame = ctk.CTkFrame(self.settings_frame)
        general_frame.pack(fill="x", padx=10, pady=5)

        # 섹션 제목
        general_title = ctk.CTkLabel(
            general_frame,
            text="일반",
            font=ctk.CTkFont(weight="bold")
        )
        general_title.pack(pady=(10, 5))

        # 자동 저장
        self.auto_save_var = ctk.BooleanVar(value=self.settings["auto_save"])
        auto_save_checkbox = ctk.CTkCheckBox(
            general_frame,
            text="대화 자동 저장",
            variable=self.auto_save_var,
            command=self.on_auto_save_change
        )
        auto_save_checkbox.pack(pady=2)

        # 알림 소리
        self.notification_var = ctk.BooleanVar(value=self.settings["notification_sound"])
        notification_checkbox = ctk.CTkCheckBox(
            general_frame,
            text="알림 소리",
            variable=self.notification_var,
            command=self.on_notification_change
        )
        notification_checkbox.pack(pady=2)

        # 설정 초기화 버튼
        reset_button = ctk.CTkButton(
            general_frame,
            text="설정 초기화",
            command=self.reset_settings,
            height=30,
            fg_color="red",
            hover_color="darkred"
        )
        reset_button.pack(pady=10)

    def on_mode_change(self, value):
        """외관 모드 변경"""
        self.settings["appearance_mode"] = value
        self.change_callback("appearance_mode", value)

    def on_theme_change(self, value):
        """색상 테마 변경"""
        self.settings["color_theme"] = value
        self.change_callback("color_theme", value)

    def on_learning_change(self):
        """학습 설정 변경"""
        value = self.learning_var.get()
        self.settings["enable_learning"] = value
        self.change_callback("enable_learning", value)

    def on_korean_nlp_change(self):
        """한국어 NLP 설정 변경"""
        value = self.korean_nlp_var.get()
        self.settings["enable_korean_nlp"] = value
        self.change_callback("enable_korean_nlp", value)

    def on_timeout_change(self, value):
        """응답 시간 제한 변경"""
        timeout_value = float(value)
        self.settings["response_timeout"] = timeout_value
        self.timeout_value_label.configure(text=f"{timeout_value:.1f}s")
        self.change_callback("response_timeout", timeout_value)

    def on_quality_change(self, value):
        """품질 임계값 변경"""
        quality_value = float(value)
        self.settings["quality_threshold"] = quality_value
        self.quality_value_label.configure(text=f"{quality_value:.1f}")
        self.change_callback("quality_threshold", quality_value)

    def on_auto_save_change(self):
        """자동 저장 설정 변경"""
        value = self.auto_save_var.get()
        self.settings["auto_save"] = value
        self.change_callback("auto_save", value)

    def on_notification_change(self):
        """알림 소리 설정 변경"""
        value = self.notification_var.get()
        self.settings["notification_sound"] = value
        self.change_callback("notification_sound", value)

    def apply_appearance_settings(self):
        """외관 설정 적용"""
        # CustomTkinter 설정 적용
        ctk.set_appearance_mode(self.settings["appearance_mode"])
        ctk.set_default_color_theme(self.settings["color_theme"])

        # 성공 메시지 (임시 라벨)
        success_label = ctk.CTkLabel(
            self.settings_frame,
            text="외관 설정이 적용되었습니다!",
            text_color="green"
        )
        success_label.pack(pady=5)

        # 3초 후 메시지 제거
        self.parent.after(3000, success_label.destroy)

    def reset_settings(self):
        """설정 초기화"""
        # 기본값으로 재설정
        default_settings = {
            "appearance_mode": "dark",
            "color_theme": "blue",
            "enable_learning": True,
            "enable_korean_nlp": True,
            "response_timeout": 5.0,
            "quality_threshold": 0.7,
            "auto_save": True,
            "notification_sound": True
        }

        self.settings.update(default_settings)

        # UI 업데이트
        self.mode_var.set(default_settings["appearance_mode"])
        self.theme_var.set(default_settings["color_theme"])
        self.learning_var.set(default_settings["enable_learning"])
        self.korean_nlp_var.set(default_settings["enable_korean_nlp"])
        self.timeout_var.set(default_settings["response_timeout"])
        self.quality_var.set(default_settings["quality_threshold"])
        self.auto_save_var.set(default_settings["auto_save"])
        self.notification_var.set(default_settings["notification_sound"])

        # 라벨 업데이트
        self.timeout_value_label.configure(text=f"{default_settings['response_timeout']:.1f}s")
        self.quality_value_label.configure(text=f"{default_settings['quality_threshold']:.1f}")

        # 콜백 호출
        for key, value in default_settings.items():
            self.change_callback(key, value)

        # 성공 메시지
        success_label = ctk.CTkLabel(
            self.settings_frame,
            text="설정이 초기화되었습니다!",
            text_color="orange"
        )
        success_label.pack(pady=5)
        self.parent.after(3000, success_label.destroy)

    def get_settings(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self.settings.copy()

    def load_settings(self, settings: Dict[str, Any]):
        """설정 로드"""
        self.settings.update(settings)

        # UI 업데이트
        self.mode_var.set(self.settings["appearance_mode"])
        self.theme_var.set(self.settings["color_theme"])
        self.learning_var.set(self.settings["enable_learning"])
        self.korean_nlp_var.set(self.settings["enable_korean_nlp"])
        self.timeout_var.set(self.settings["response_timeout"])
        self.quality_var.set(self.settings["quality_threshold"])
        self.auto_save_var.set(self.settings["auto_save"])
        self.notification_var.set(self.settings["notification_sound"])

        # 라벨 업데이트
        self.timeout_value_label.configure(text=f"{self.settings['response_timeout']:.1f}s")
        self.quality_value_label.configure(text=f"{self.settings['quality_threshold']:.1f}")