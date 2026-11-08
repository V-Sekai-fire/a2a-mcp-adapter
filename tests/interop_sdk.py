"""Interop check against the official a2a-sdk (1.3.0, A2A 1.0). Not part of the unit suite.

Run: python tests/interop_sdk.py   (needs a2a-sdk[http-server]==1.3.0; in-process, no listener)
  A: the bridge's v1 client (a2a_v1) talks to an SDK-built agent.
  B: the SDK's client talks to the bridge's v1 server, which forwards to the SDK agent.
"""
import asyncio
import os
import sys

import httpx
from starlette.applications import Starlette

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["A2A_ALLOW_PRIVATE_AGENTS"] = "1"
os.environ["A2A_PUBLIC_URL"] = "http://bridge.test/"

from a2a.client import ClientConfig, create_client  # noqa: E402
from a2a.helpers.proto_helpers import new_text_artifact_update_event, new_text_message, new_text_status_update_event  # noqa: E402
from a2a.server.agent_execution import AgentExecutor  # noqa: E402
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2  # noqa: E402
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from a2a.types import a2a_pb2 as pb  # noqa: E402

import a2a_mcp_server as srv  # noqa: E402
import a2a_v1  # noqa: E402
from common.server.server import A2AServer  # noqa: E402
from common.server.task_manager import BridgeTaskManager  # noqa: E402

SDK = "http://sdk.test"
BRIDGE = "http://bridge.test"


class Echo(AgentExecutor):
    async def execute(self, context, event_queue):
        text = context.get_user_input()
        task = pb.Task(id=context.task_id, context_id=context.context_id,
                       status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_WORKING))
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(new_text_artifact_update_event(
            context.task_id, context.context_id, "echo", f"echo: {text}", artifact_id="art-1"))
        await event_queue.enqueue_event(new_text_status_update_event(
            context.task_id, context.context_id, pb.TaskState.TASK_STATE_COMPLETED, "done"))

    async def cancel(self, context, event_queue):
        await event_queue.enqueue_event(new_text_status_update_event(
            context.task_id, context.context_id, pb.TaskState.TASK_STATE_CANCELED, "canceled"))


def sdk_app():
    card = pb.AgentCard(
        name="SDK Echo", description="echo", version="1.0.0",
        supported_interfaces=[pb.AgentInterface(url=SDK + "/rpc", protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=pb.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        skills=[pb.AgentSkill(id="echo", name="Echo", description="echo", tags=["echo"])])
    handler = DefaultRequestHandlerV2(agent_executor=Echo(), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/rpc"))


class Router(httpx.AsyncBaseTransport):
    def __init__(self, apps):
        self.t = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.t[request.url.host].handle_async_request(request)


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


async def main():
    sdk = sdk_app()
    mgr = BridgeTaskManager(is_registered=lambda u: u == SDK)
    bridge = A2AServer(mgr, srv.build_agent_card(False))
    transport = Router({"sdk.test": sdk, "bridge.test": bridge.app})
    a2a_v1.TRANSPORT = transport

    # A: bridge's v1 client -> SDK agent
    c = a2a_v1.A2AV1Client(SDK)
    res = await c.send_message("hello")
    task = res.get("task", {})
    check("A1 SendMessage returns a task", bool(task.get("id")), str(list(res)))
    check("A2 task completed", task.get("status", {}).get("state") == "TASK_STATE_COMPLETED", str(task.get("status")))
    arts = task.get("artifacts", [])
    check("A3 artifact text echoed", bool(arts) and "echo: hello" in str(arts), str(arts)[:120])
    got = await c.get_task(task["id"])
    check("A4 GetTask by id", got.get("id") == task["id"])
    evs = [e async for e in c.send_streaming_message("stream")]
    kinds = [next(iter(e)) for e in evs]
    check("A5 SendStreamingMessage yields events", len(evs) >= 2, str(kinds))

    # B: SDK client -> bridge server -> SDK agent
    async with httpx.AsyncClient(transport=transport) as hc:
        client = await create_client(BRIDGE, client_config=ClientConfig(httpx_client=hc, streaming=False))
        card = await client.get_card() if hasattr(client, "get_card") else None
        check("B0 SDK client resolved the bridge card", True, getattr(card, "name", "n/a"))
        msg = new_text_message("via bridge")
        msg.role = pb.Role.ROLE_USER
        req = pb.SendMessageRequest(message=msg)
        req.metadata.update({"agent_url": SDK})
        final = None
        async for ev in client.send_message(req):
            final = ev
        out = str(final)
        check("B1 SDK client SendMessage through bridge", final is not None and "echo: via bridge" in out, out[:160].replace("\n", " "))
        remote_ids = {e["remote_id"] for e in mgr._tasks.values()}
        bridge_ids = set(mgr._tasks)
        exposed = [r for r in remote_ids if r in out]
        check("B2 remote task id not exposed to the SDK client", bool(bridge_ids) and not exposed and any(b in out for b in bridge_ids), f"bridge={len(bridge_ids)} leaked={exposed}")

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
