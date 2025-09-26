"""
Desktop Application UI Components
PACA v5 Desktop Application
"""

from .chat_interface import ChatInterface, ChatMessage
from .settings_panel import SettingsPanel
from .status_bar import StatusBar

__all__ = [
    "ChatInterface",
    "ChatMessage",
    "SettingsPanel",
    "StatusBar"
]