import types

import pytest

from paca.system import PacaSystem, PacaConfig


@pytest.mark.asyncio
async def test_event_bus_links_subsystems_and_updates_metrics():
    system = PacaSystem(PacaConfig())

    system.cognitive_system = types.SimpleNamespace(
        events=None,
        processors={"default": types.SimpleNamespace(events=None)}
    )
    system.reasoning_engine = types.SimpleNamespace(
        events=None,
        engines={"deductive": types.SimpleNamespace(events=None)}
    )
    service_stub = types.SimpleNamespace(events=None, config=types.SimpleNamespace(id="svc", name="svc"))
    system.service_manager = types.SimpleNamespace(events=None, services={"svc": service_stub})

    await system._setup_event_handlers()

    assert system.cognitive_system.events is system.event_bus
    assert system.cognitive_system.processors["default"].events is system.event_bus
    assert system.reasoning_engine.events is system.event_bus
    assert system.reasoning_engine.engines["deductive"].events is system.event_bus
    assert system.service_manager.events is system.event_bus
    assert service_stub.events is system.event_bus

    await system.event_bus.emit('cognitive.process.completed', {
        'processor_id': 'default',
        'confidence': 0.82,
        'processing_time_ms': 110,
    })
    await system.event_bus.emit('cognitive.process.failed', {'error': 'validation failed'})
    await system.event_bus.emit('reasoning.completed', {
        'confidence': 0.91,
        'execution_time_ms': 48,
    })
    await system.event_bus.emit('reasoning.failed', {'error': 'timeout'})
    await system.event_bus.emit('service.started', {'service_name': 'svc'})
    await system.event_bus.emit('service.start_failed', {'service_name': 'svc'})

    metrics = system.performance_metrics

    assert metrics['cognitive_events']['completed'] == 1
    assert metrics['cognitive_events']['failed'] == 1
    assert metrics['cognitive_events']['last_confidence'] == 0.82
    assert metrics['cognitive_events']['last_error'] == 'validation failed'
    assert metrics['reasoning_events']['completed'] == 1
    assert metrics['reasoning_events']['failed'] == 1
    assert metrics['reasoning_events']['last_confidence'] == 0.91
    assert metrics['reasoning_events']['last_error'] == 'timeout'
    assert metrics['service_events']['failed'] == ['svc']
    assert metrics['service_events']['started'] == []
