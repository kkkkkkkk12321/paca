#!/usr/bin/env python3
"""
PACA v5 Phase 4 Simple Integration Test
UI/UX System Basic Functionality Test
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent / "paca_python"
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Basic import test"""
    print("=== Basic Import Test ===")

    try:
        # Enhanced GUI imports
        from desktop_app.enhanced_gui import (
            EnhancedGUI, ChatInterface, MonitoringPanel, BackupManager
        )
        print("SUCCESS: Enhanced GUI imports successful")

        # Debug Panel imports
        from desktop_app.debug_panel import (
            DebugPanel, ReasoningDisplay, ComplexityAnalyzer
        )
        print("SUCCESS: Debug Panel imports successful")

        return True
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return False

def test_enhanced_gui_creation():
    """Enhanced GUI creation test"""
    print("\n=== Enhanced GUI Creation Test ===")

    try:
        from desktop_app.enhanced_gui import EnhancedGUI

        # Test GUI creation without running mainloop
        gui = EnhancedGUI()
        print("SUCCESS: EnhancedGUI created")

        # Check components
        if gui.chat_interface:
            print("SUCCESS: ChatInterface initialized")
        else:
            print("WARNING: ChatInterface not initialized")

        if gui.monitoring_panel:
            print("SUCCESS: MonitoringPanel initialized")
        else:
            print("WARNING: MonitoringPanel not initialized")

        if gui.backup_manager:
            print("SUCCESS: BackupManager initialized")
        else:
            print("WARNING: BackupManager not initialized")

        # Test learning systems
        if hasattr(gui, 'iis_calculator'):
            print("SUCCESS: Learning systems initialized")
        else:
            print("WARNING: Learning systems not initialized")

        # Clean up
        gui.root.destroy()
        print("SUCCESS: GUI cleanup completed")

        return True

    except Exception as e:
        print(f"FAILED: Enhanced GUI test failed: {e}")
        return False

def test_debug_panel_creation():
    """Debug Panel creation test"""
    print("\n=== Debug Panel Creation Test ===")

    try:
        from desktop_app.debug_panel import DebugPanel, DebugLevel

        # Test debug panel creation in non-standalone mode
        debug_panel = DebugPanel(standalone=False)
        print("SUCCESS: DebugPanel created (non-standalone)")

        # Test debug entry addition
        debug_panel.add_debug_entry(
            DebugLevel.INFO,
            "TEST",
            "Test debug entry"
        )
        print("SUCCESS: Debug entry added")

        # Check debug entries
        if debug_panel.debug_entries:
            entry = debug_panel.debug_entries[0]
            print(f"SUCCESS: Debug entry recorded - Level: {entry.level.value}, Message: {entry.message}")
        else:
            print("WARNING: No debug entries found")

        # Test AI systems
        if debug_panel.complexity_detector:
            print("SUCCESS: ComplexityDetector initialized")
        else:
            print("WARNING: ComplexityDetector not initialized")

        if debug_panel.reasoning_chain:
            print("SUCCESS: ReasoningChain initialized")
        else:
            print("WARNING: ReasoningChain not initialized")

        return True

    except Exception as e:
        print(f"FAILED: Debug Panel test failed: {e}")
        return False

def test_component_integration():
    """Component integration test"""
    print("\n=== Component Integration Test ===")

    try:
        from desktop_app.enhanced_gui import ChatInterface, MonitoringPanel
        from desktop_app.debug_panel import ComplexityAnalyzer, ReasoningDisplay
        import tkinter as tk

        # Create a test root window
        root = tk.Tk()
        root.withdraw()  # Hide the window

        # Test ChatInterface
        chat_frame = tk.Frame(root)
        chat_interface = ChatInterface(chat_frame)
        print("SUCCESS: ChatInterface component created")

        # Test MonitoringPanel
        monitor_frame = tk.Frame(root)
        monitoring_panel = MonitoringPanel(monitor_frame)
        print("SUCCESS: MonitoringPanel component created")

        # Test ComplexityAnalyzer
        complexity_frame = tk.Frame(root)
        complexity_analyzer = ComplexityAnalyzer(complexity_frame)
        print("SUCCESS: ComplexityAnalyzer component created")

        # Test ReasoningDisplay
        reasoning_frame = tk.Frame(root)
        reasoning_display = ReasoningDisplay(reasoning_frame)
        print("SUCCESS: ReasoningDisplay component created")

        # Test callback functionality
        def test_callback(message):
            print(f"CALLBACK: Message received - {message.sender}")

        chat_interface.add_message_callback(test_callback)
        print("SUCCESS: Callback registration works")

        # Clean up
        root.destroy()
        print("SUCCESS: Component cleanup completed")

        return True

    except Exception as e:
        print(f"FAILED: Component integration test failed: {e}")
        return False

def test_learning_system_integration():
    """Learning system integration test"""
    print("\n=== Learning System Integration Test ===")

    try:
        from desktop_app.enhanced_gui import LearningStatus
        from desktop_app.debug_panel import DebugLevel

        # Test LearningStatus dataclass
        learning_status = LearningStatus(
            iis_score=75,
            trend="improving",
            tactics_count=23,
            heuristics_count=15,
            recent_improvements=["Test improvement"]
        )
        print("SUCCESS: LearningStatus created")
        print(f"  IIS Score: {learning_status.iis_score}")
        print(f"  Trend: {learning_status.trend}")
        print(f"  Tactics: {learning_status.tactics_count}")

        # Test DebugLevel enum
        for level in DebugLevel:
            print(f"  Debug level: {level.value}")
        print("SUCCESS: DebugLevel enum accessible")

        return True

    except Exception as e:
        print(f"FAILED: Learning system integration test failed: {e}")
        return False

def test_ai_system_integration():
    """AI system integration test"""
    print("\n=== AI System Integration Test ===")

    try:
        # Import AI systems used by GUI components
        from paca.cognitive import ComplexityDetector, ReasoningChain, MetacognitionEngine
        from paca.learning import IISCalculator, TacticGenerator
        from paca.performance import HardwareMonitor, ProfileManager

        # Test system creation
        complexity_detector = ComplexityDetector()
        print("SUCCESS: ComplexityDetector created")

        reasoning_chain = ReasoningChain()
        print("SUCCESS: ReasoningChain created")

        metacognition = MetacognitionEngine()
        print("SUCCESS: MetacognitionEngine created")

        iis_calculator = IISCalculator()
        print("SUCCESS: IISCalculator created")

        tactic_generator = TacticGenerator()
        print("SUCCESS: TacticGenerator created")

        hardware_monitor = HardwareMonitor()
        print("SUCCESS: HardwareMonitor created")

        profile_manager = ProfileManager()
        print("SUCCESS: ProfileManager created")

        print("SUCCESS: All AI systems integrated properly")
        return True

    except Exception as e:
        print(f"FAILED: AI system integration test failed: {e}")
        return False

def test_performance_metrics():
    """Performance metrics test"""
    print("\n=== Performance Metrics Test ===")

    try:
        from desktop_app.enhanced_gui import EnhancedGUI
        import time

        print("PERFORMANCE: Testing GUI creation time...")

        start_time = time.time()
        gui = EnhancedGUI()
        end_time = time.time()

        creation_time = (end_time - start_time) * 1000
        print(f"  GUI creation time: {creation_time:.1f}ms")

        # Test component access time
        start_time = time.time()
        chat = gui.chat_interface
        monitor = gui.monitoring_panel
        backup = gui.backup_manager
        end_time = time.time()

        access_time = (end_time - start_time) * 1000
        print(f"  Component access time: {access_time:.1f}ms")

        # Clean up
        gui.root.destroy()

        # Performance validation
        if creation_time <= 5000:  # 5 seconds
            print("SUCCESS: GUI creation within performance target")
        else:
            print(f"WARNING: GUI creation slow ({creation_time:.1f}ms > 5000ms)")

        return True

    except Exception as e:
        print(f"FAILED: Performance metrics test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("PACA v5 Phase 4 UI/UX System Integration Test")
    print("=" * 60)

    # Test list
    tests = [
        ("Basic Import", test_basic_imports),
        ("Enhanced GUI Creation", test_enhanced_gui_creation),
        ("Debug Panel Creation", test_debug_panel_creation),
        ("Component Integration", test_component_integration),
        ("Learning System Integration", test_learning_system_integration),
        ("AI System Integration", test_ai_system_integration),
        ("Performance Metrics", test_performance_metrics),
    ]

    passed_tests = 0
    total_tests = len(tests)

    # Run tests
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"PASS: {test_name} test passed")
            else:
                print(f"FAIL: {test_name} test failed")
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {e}")

    # Final results
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("SUCCESS: All tests passed successfully!")
        print("Phase 4 UI/UX System is working correctly!")
    else:
        print(f"WARNING: {total_tests - passed_tests} tests failed.")
        print("Some features may have issues.")

    return passed_tests == total_tests

if __name__ == "__main__":
    # Run tests
    success = main()
    sys.exit(0 if success else 1)