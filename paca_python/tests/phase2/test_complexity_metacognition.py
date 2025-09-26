import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paca.cognitive import (
    ComplexityDetector,
    MetacognitionEngine,
    QualityLevel,
)


@pytest.mark.asyncio
async def test_complexity_detector_uses_external_config_and_cache_behavior():
    detector = ComplexityDetector(
        {
            "complexity": {
                "reasoning_threshold": 35,
                "feature_weights": {"keyword": 0.4, "structure": 0.2, "domain": 0.2, "reasoning": 0.2},
                "cache": {"enabled": True, "ttl_seconds": 999, "max_entries": 4},
            }
        }
    )

    text = """만약 사용자가 복잡한 시스템 설계를 요청하고 단계별 비교를 원한다면 어떻게 대응해야 할까요?"""

    first_result = await detector.detect_complexity(text)
    assert first_result.reasoning_required is True
    assert first_result.analysis_details["cache_hit"] is False
    assert first_result.analysis_details["features"]["question_count"] >= 1
    assert first_result.analysis_details["reasons"], "설명 근거가 포함되어야 합니다"
    assert pytest.approx(sum(detector.feature_weights.values()), 0.001) == 1.0

    cached_result = await detector.detect_complexity(text)
    assert cached_result.score == first_result.score
    assert cached_result.analysis_details["cache_hit"] is True


@pytest.mark.asyncio
async def test_metacognition_engine_generates_quality_assessment_and_alerts(tmp_path: Path):
    log_dir = tmp_path / "metacog"
    engine = MetacognitionEngine(
        {
            "metacognition": {
                "quality_thresholds": {"green": 90, "yellow": 55, "red": 40},
                "alerts": {"high_complexity_score": 60, "low_confidence": 0.55},
                "logging_enabled": True,
                "log_directory": str(log_dir),
                "max_history_size": 10,
            }
        }
    )

    session_id = await engine.start_reasoning_monitoring(
        {
            "user_id": "tester",
            "message": "복잡한 전략 정리",
            "complexity_score": 78,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    await engine.add_reasoning_step(
        session_id=session_id,
        step_description="1. 문제 정의",
        input_data={"premise": "고객 요구"},
        output_data={"conclusion": "요구 분석", "confidence": 0.5},
        processing_time_ms=1500,
    )
    await engine.add_reasoning_step(
        session_id=session_id,
        step_description="2. 대안 비교",
        input_data={"previous": "요구 분석"},
        output_data={"conclusion": "전략 선택", "confidence": 0.4},
        processing_time_ms=1800,
    )

    session = await engine.end_monitoring_session(session_id)

    assert session.quality_assessment is not None
    assert session.quality_assessment.level in {QualityLevel.YELLOW, QualityLevel.RED}
    assert session.alerts, "경보 정보가 기록되어야 합니다"

    log_files = list(log_dir.glob("*.log"))
    assert log_files, "메타인지 로그 파일이 생성되어야 합니다"
    with log_files[0].open(encoding="utf-8") as handle:
        payload = json.loads(handle.readline())
    assert payload["session_id"] == session.session_id
    assert payload["quality"]["level"] == session.quality_assessment.level.value
