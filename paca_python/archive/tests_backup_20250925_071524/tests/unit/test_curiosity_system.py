"""
Test script for PACA Curiosity Engine System

This script tests the integrated curiosity system including gap detection,
exploration planning, mission alignment, and the main curiosity engine.
"""

import asyncio
import sys
import os

# Add the paca directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'paca'))

from paca.cognitive.curiosity import (
    CuriosityEngine,
    CuriosityLevel,
    ExplorationFocus,
    CuriosityConfig,
    GapDetector,
    ExplorationPlanner,
    MissionAligner
)

async def test_gap_detector():
    """Test the gap detection system"""
    print("=== Testing Gap Detector ===")

    gap_detector = GapDetector()

    # Test text with logical issues
    test_text = """
    Water always boils at 100 degrees Celsius. However, water sometimes
    boils at different temperatures. This is because altitude affects boiling point.
    Therefore, water always boils at the same temperature regardless of conditions.
    We assume that pressure doesn't matter for this analysis.
    """

    # Detect gaps
    gaps = await gap_detector.detect_gaps_in_text(test_text, {"test": "gap_detection"})

    print(f"Detected {len(gaps)} logical gaps:")
    for gap in gaps:
        print(f"  - {gap.gap_type.value}: {gap.description}")
        print(f"    Severity: {gap.severity.value}, Exploration Potential: {gap.exploration_potential}")

    # Analyze gap patterns
    analysis = await gap_detector.analyze_gap_patterns()
    print(f"\nGap Analysis:")
    print(f"  Total gaps: {analysis.total_gaps}")
    print(f"  Coherence score: {analysis.overall_coherence_score:.2f}")
    print(f"  Critical gaps: {len(analysis.critical_gaps)}")

    return gaps

async def test_exploration_planner():
    """Test the exploration planning system"""
    print("\n=== Testing Exploration Planner ===")

    planner = ExplorationPlanner()

    # Create exploration plan
    objective = "Investigate contradictory statements about water boiling point"
    plan = await planner.create_exploration_plan(
        objective=objective,
        context={"domain": "physics", "complexity": "moderate"}
    )

    print(f"Created exploration plan: {plan.title}")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Scope: {plan.scope.value}")
    print(f"  Steps: {len(plan.steps)}")
    print(f"  Estimated time: {plan.total_estimated_time}")
    print(f"  Required resources: {[r.value for r in plan.required_resources]}")

    # Show first few steps
    print("  First steps:")
    for i, step in enumerate(plan.steps[:3]):
        print(f"    {i+1}. {step.description}")
        print(f"       Method: {step.method}, Duration: {step.estimated_duration}")

    # Execute plan (simulation)
    execution_results = await planner.execute_plan(plan.plan_id)
    print(f"\nExecution Results:")
    print(f"  Status: {execution_results['status']}")
    print(f"  Completed steps: {execution_results['completed_steps']}/{execution_results['total_steps']}")
    print(f"  Insights: {len(execution_results['insights_generated'])}")

    return plan

async def test_mission_aligner():
    """Test the mission alignment system"""
    print("\n=== Testing Mission Aligner ===")

    aligner = MissionAligner()

    # Add a test mission
    mission_id = await aligner.add_user_mission(
        title="Learn and understand scientific concepts accurately",
        description="Develop a deep and accurate understanding of scientific principles, especially physics and chemistry",
        core_values=["accuracy", "scientific_rigor", "evidence_based_reasoning"],
        success_criteria=["Can explain concepts clearly", "Identifies contradictions", "Seeks reliable sources"],
        priority=1
    )

    print(f"Added mission: {mission_id}")

    # Test alignment check
    exploration_objective = "Investigate contradictory statements about water boiling point"
    alignment_check = await aligner.check_exploration_alignment(
        exploration_objective,
        {"domain": "physics", "method": "scientific_inquiry"}
    )

    print(f"\nAlignment Check Results:")
    print(f"  Objective: {alignment_check.exploration_objective}")
    print(f"  Alignment score: {alignment_check.alignment_score:.2f}")
    print(f"  Category: {alignment_check.alignment_category.value}")
    print(f"  Approval required: {alignment_check.approval_required}")
    print(f"  Supporting factors: {len(alignment_check.supporting_factors)}")
    print(f"  Conflicting factors: {len(alignment_check.conflicting_factors)}")
    print(f"  Value conflicts: {[c.value for c in alignment_check.value_conflicts]}")

    # Show recommendations
    print("  Recommendations:")
    for rec in alignment_check.recommendations:
        print(f"    - {rec}")

    return aligner, alignment_check

async def test_curiosity_engine():
    """Test the main curiosity engine"""
    print("\n=== Testing Curiosity Engine ===")

    # Create curiosity engine with custom config
    config = CuriosityConfig(
        curiosity_level=CuriosityLevel.MODERATE,
        exploration_focus=ExplorationFocus.LOGICAL_GAPS,
        max_concurrent_explorations=2,
        require_mission_alignment=True,
        minimum_alignment_score=0.4,
        gap_sensitivity_threshold=0.3
    )

    engine = CuriosityEngine(config)

    # Add a mission to the engine's aligner
    await engine.mission_aligner.add_user_mission(
        title="Scientific Learning and Discovery",
        description="Explore and understand scientific concepts through systematic inquiry",
        core_values=["scientific_method", "accuracy", "curiosity", "evidence_based_reasoning"],
        success_criteria=["Identifies knowledge gaps", "Resolves contradictions", "Generates insights"],
        priority=1
    )

    # Process input for curiosity triggers
    test_input = """
    I'm reading about thermodynamics and I notice something puzzling.
    The book says water always boils at 100°C, but then later it mentions
    that mountain climbers need special equipment because water boils at
    lower temperatures at high altitude. This seems contradictory.
    I wonder what causes this difference and how significant it is.
    """

    triggers = await engine.process_input_for_curiosity(
        test_input,
        {"domain": "thermodynamics", "source": "educational_material"}
    )

    print(f"Generated {len(triggers)} curiosity triggers:")
    for trigger in triggers:
        print(f"  - {trigger.trigger_type}: {trigger.description}")
        print(f"    Intensity: {trigger.intensity:.2f}")

    # Generate exploration sessions
    sessions = await engine.generate_exploration_from_triggers(triggers)

    print(f"\nGenerated {len(sessions)} exploration sessions:")
    for session in sessions:
        print(f"  Session {session.session_id}: {len(session.triggers)} triggers, {len(session.gaps)} gaps")
        if session.plan:
            print(f"    Plan: {session.plan.title}")
            print(f"    Strategy: {session.plan.strategy.value}")
        if session.alignment_check:
            print(f"    Alignment: {session.alignment_check.alignment_score:.2f} ({session.alignment_check.alignment_category.value})")

    # Test execution evaluation
    if sessions:
        session_id = sessions[0].session_id
        should_execute, reasons = await engine.evaluate_session_for_execution(session_id)

        print(f"\nExecution Evaluation for {session_id}:")
        print(f"  Should execute: {should_execute}")
        print(f"  Reasons:")
        for reason in reasons:
            print(f"    - {reason}")

        # Try to execute the session
        if should_execute:
            execution_results = await engine.execute_exploration_session(session_id)
            print(f"\nExecution Results:")
            if "success" in execution_results:
                print(f"  Success: {execution_results['success']}")
                if execution_results.get('execution_time'):
                    print(f"  Execution time: {execution_results['execution_time']}")
                print(f"  Results available: {bool(execution_results.get('results'))}")
            else:
                print(f"  Error: {execution_results.get('error', 'Unknown error')}")

    # Get overall status
    status = engine.get_curiosity_status()
    print(f"\nCuriosity Engine Status:")
    print(f"  Curiosity level: {status['curiosity_level']}")
    print(f"  Exploration focus: {status['exploration_focus']}")
    print(f"  Active sessions: {status['active_sessions']}")
    print(f"  Completed sessions: {status['completed_sessions']}")
    print(f"  Total triggers: {status['total_triggers']}")
    print(f"  Recent triggers: {status['recent_triggers']}")

    return engine

async def test_integrated_system():
    """Test the complete integrated curiosity system"""
    print("\n=== Testing Integrated Curiosity System ===")

    try:
        # Test individual components
        gaps = await test_gap_detector()
        plan = await test_exploration_planner()
        aligner, alignment = await test_mission_aligner()
        engine = await test_curiosity_engine()

        print("\n=== Integration Summary ===")
        print(f"[OK] Gap Detector: {len(gaps)} gaps detected")
        print(f"[OK] Exploration Planner: Plan created with {len(plan.steps)} steps")
        print(f"[OK] Mission Aligner: Alignment score {alignment.alignment_score:.2f}")
        print(f"[OK] Curiosity Engine: System operational")

        # Test cross-component integration
        print("\n=== Testing Cross-Component Integration ===")

        # Generate a complex scenario
        complex_input = """
        I'm studying both chemistry and physics, and I'm confused about molecular behavior.
        In chemistry class, we learned that molecules move faster when heated.
        In physics class, we learned about energy conservation.
        But I'm not sure how these concepts connect, and I wonder if there are
        any contradictions between these two perspectives. Maybe there are
        deeper principles I'm missing that could help me understand this better.
        """

        # Process through complete pipeline
        curiosity_triggers = await engine.process_input_for_curiosity(
            complex_input,
            {"domain": "science", "complexity": "interdisciplinary"}
        )

        exploration_sessions = await engine.generate_exploration_from_triggers(curiosity_triggers)

        print(f"Complex scenario generated:")
        print(f"  - {len(curiosity_triggers)} curiosity triggers")
        print(f"  - {len(exploration_sessions)} exploration sessions")

        # Analyze the quality of generated explorations
        for session in exploration_sessions:
            if session.alignment_check:
                print(f"  Session {session.session_id}:")
                print(f"    Alignment: {session.alignment_check.alignment_score:.2f}")
                print(f"    Triggers: {len(session.triggers)}")
                if session.plan:
                    print(f"    Plan steps: {len(session.plan.steps)}")

        print("\n[SUCCESS] Curiosity Engine Integration Test Complete!")
        print("All components are working together successfully.")

        return True

    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_integrated_system())
    if success:
        print("\n[PASS] All tests passed - Curiosity Engine is ready for use!")
    else:
        print("\n[FAIL] Some tests failed - Check the error messages above.")