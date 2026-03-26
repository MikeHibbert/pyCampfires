# Extending Campfires: Contracts and Examples

This document summarizes new extension contracts intended to keep the core small while enabling advanced capabilities in downstream systems like Campfire Valley.

## Event Bus

```python
import asyncio
from campfires import EventBus

async def main():
    bus = EventBus()
    await bus.start()
    bus.subscribe("torch_sent", lambda e: print(e.payload))
    await bus.publish("torch_sent", {"campfire": "example", "torch_id": "t1"})
    await bus.stop()

asyncio.run(main())
```

## Security Hooks

Security hooks allow allow/transform/reject decisions for inbound/outbound torches.

```python
from campfires import SecurityHooks, SecurityHookResult

class MyHooks(SecurityHooks):
    async def pre_receive_torch(self, torch, context):
        if "blocked" in torch.claim:
            return SecurityHookResult(action="reject", reason="blocked_content")
        return SecurityHookResult(action="allow", torch=torch)
```

## Routing Strategy

```python
from campfires import RoutingStrategy, TargetDecision

class FixedChannel(RoutingStrategy):
    async def choose_target(self, torch, session=None):
        return TargetDecision(channel="campfire.my-target.in")
```

## Session Manager

```python
from campfires import InMemorySessionManager

sm = InMemorySessionManager()
sess = asyncio.run(sm.create("sid-123", {"user": "u1"}))
```

## Tool Adapter and Scheduler

```python
from campfires import ToolAdapter, AsyncScheduler

class EchoTool(ToolAdapter):
    async def invoke(self, name, params):
        return {"ok": True, "name": name, "params": params}

sched = AsyncScheduler()
async def tick():
    print("tick")

asyncio.run(sched.register_interval("heartbeat", 5, tick))
```

## Integrating with Campfire

```python
from campfires import Campfire, SecurityHooks, EventBus, InMemorySessionManager

campfire = Campfire(
    name="example",
    campers=[],
    party_box=...,
    event_bus=EventBus(),
    security_hooks=SecurityHooks(),
    session_manager=InMemorySessionManager()
)
```

These contracts are optional; defaults are provided to preserve backward compatibility. Downstream systems implement concrete adapters and strategies and inject them into Campfires at construction time.
*** End Patch***} وٺabadde -->
