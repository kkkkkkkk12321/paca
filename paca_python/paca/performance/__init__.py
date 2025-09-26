"""
Performance Module - PACA Python v5
실시간 하드웨어 모니터링 및 성능 최적화 시스템
"""

from .hardware_monitor import (
    HardwareMonitor,
    SystemStatus,
    PerformanceMetrics,
    ResourceUsage,
    PerformanceAlert
)

from .profile_manager import (
    PerformanceProfile,
    ProfileManager,
    ProfileType,
    ProfileConfig
)

__all__ = [
    # Hardware Monitoring
    'HardwareMonitor',
    'SystemStatus',
    'PerformanceMetrics',
    'ResourceUsage',
    'PerformanceAlert',

    # Profile Management
    'PerformanceProfile',
    'ProfileManager',
    'ProfileType',
    'ProfileConfig'
]

__version__ = "5.0.0"
__author__ = "PACA Development Team"