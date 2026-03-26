import asyncio
from pathlib import Path
import importlib.util
import pytest

_ROOT = Path(__file__).resolve().parents[1]

def _load(name: str, rel: str):
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

event_bus_mod = _load("event_bus", "campfires/core/event_bus.py")
security_mod = _load("security_hooks", "campfires/core/security_hooks.py")
session_mod = _load("session_manager", "campfires/core/session_manager.py")
torch_mod = _load("torch", "campfires/core/torch.py")

EventBus = event_bus_mod.EventBus
SecurityHooks = security_mod.SecurityHooks
InMemorySessionManager = session_mod.InMemorySessionManager
Torch = torch_mod.Torch


@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = EventBus()
    received = asyncio.Event()

    def handler(event):
        if event.event_type == "ping" and event.payload.get("x") == 1:
            received.set()

    bus.subscribe("ping", handler)
    await bus.start()
    try:
        await bus.publish("ping", {"x": 1})
        await asyncio.wait_for(received.wait(), timeout=1.0)
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_security_hooks_default_allow():
    hooks = SecurityHooks()
    t = Torch(claim="ok", source_campfire="test", channel="test")
    res_in = await hooks.pre_receive_torch(t, {"k": "v"})
    res_out = await hooks.pre_send_torch(t, {"k": "v"})
    assert res_in.action == "allow"
    assert res_out.action == "allow"


@pytest.mark.asyncio
async def test_inmemory_session_manager():
    sm = InMemorySessionManager()
    s = await sm.create("sid-1", {"a": 1})
    assert s.session_id == "sid-1"
    s2 = await sm.get("sid-1")
    assert s2 is not None and s2.data["a"] == 1
    await sm.update("sid-1", {"b": 2})
    s3 = await sm.get("sid-1")
    assert s3.data["a"] == 1 and s3.data["b"] == 2
    await sm.close("sid-1")
    assert await sm.get("sid-1") is None
