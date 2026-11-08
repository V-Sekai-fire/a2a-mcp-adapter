"""A2A v1.0 client (JSON-RPC binding) and reply rendering.

Wire format taken from https://a2a-protocol.org/latest/specification/ (1.0.0):
PascalCase methods, `contextId`, ProtoJSON enums (ROLE_*, TASK_STATE_*), parts without a
`kind` field, agent card at /.well-known/agent-card.json with `supportedInterfaces`.

Everything an agent returns is rendered through `render_*`, which wraps free text in the
untrusted envelope from security.py and passes identifiers only if they look like identifiers.
"""
import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from security import UnsafeURLError, validate_agent_url, wrap_untrusted_data, wrap_untrusted_text

A2A_VERSION = "1.0"
CARD_PATH = "/.well-known/agent-card.json"
MAX_STREAM_EVENTS = 1000
TRANSPORT: Optional[httpx.AsyncBaseTransport] = None  # tests inject a MockTransport

_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ENUM = re.compile(r"^[A-Z][A-Z0-9_]{0,40}$")


class A2AError(Exception):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


def _http(**kw) -> httpx.AsyncClient:
    # follow_redirects stays off so a public host cannot bounce us to an internal one
    return httpx.AsyncClient(follow_redirects=False, transport=TRANSPORT, **kw)


async def _check(url: str) -> str:
    try:
        return await asyncio.to_thread(validate_agent_url, url)
    except UnsafeURLError as e:
        raise A2AError(f"refusing URL: {e}")


async def fetch_card(url: str) -> Dict[str, Any]:
    """Fetch the agent card. `url` is the agent's base URL or a direct link to a .json card."""
    url = await _check(url)
    card_url = url if url.split("?")[0].endswith(".json") else url.rstrip("/") + CARD_PATH
    async with _http(timeout=10.0) as c:
        try:
            r = await c.get(card_url, headers={"A2A-Version": A2A_VERSION})
        except httpx.RequestError as e:
            raise A2AError(f"cannot fetch agent card from {card_url}: {e}")
    if r.status_code != 200:
        raise A2AError(f"agent card request answered {r.status_code}")
    try:
        card = r.json()
    except json.JSONDecodeError:
        raise A2AError("agent card is not valid JSON")
    if not isinstance(card, dict) or not isinstance(card.get("name"), str):
        raise A2AError("agent card has no name")
    return card


def pick_interface(card: Dict[str, Any]) -> Dict[str, Any]:
    for i in card.get("supportedInterfaces") or []:
        if (
            isinstance(i, dict)
            and i.get("protocolBinding") == "JSONRPC"
            and str(i.get("protocolVersion", "")).split(".")[0] == "1"
            and isinstance(i.get("url"), str)
        ):
            return i
    raise A2AError("agent card lists no JSONRPC interface for protocol 1.x")


class A2AV1Client:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout
        self._endpoint: Optional[str] = None
        self._tenant: Optional[str] = None

    async def _resolve(self) -> str:
        if self._endpoint is None:
            card = await fetch_card(self.base_url)
            iface = pick_interface(card)
            # The card is attacker-controlled, so its endpoint gets the same SSRF guard.
            self._endpoint = await _check(iface["url"])
            self._tenant = iface.get("tenant") if isinstance(iface.get("tenant"), str) else None
        return self._endpoint

    def _body(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._tenant:
            params = {**params, "tenant": self._tenant}
        return {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}

    _HEADERS = {"A2A-Version": A2A_VERSION, "Content-Type": "application/json"}

    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = await self._resolve()
        async with _http(timeout=self.timeout) as c:
            try:
                r = await c.post(endpoint, json=self._body(method, params), headers=self._HEADERS)
            except httpx.RequestError as e:
                raise A2AError(f"request failed: {e}")
        try:
            data = r.json()
        except json.JSONDecodeError:
            raise A2AError(f"agent answered {r.status_code} with non-JSON")
        if not isinstance(data, dict):
            raise A2AError("agent answered with a non-object")
        if "error" in data:
            err = data["error"] if isinstance(data["error"], dict) else {}
            raise A2AError(str(err.get("message", "agent error")), err.get("code"))
        result = data.get("result")
        if not isinstance(result, dict):
            raise A2AError("agent response has no result")
        return result

    async def stream(self, method: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        endpoint = await self._resolve()
        headers = {**self._HEADERS, "Accept": "text/event-stream"}
        async with _http(timeout=httpx.Timeout(self.timeout, read=None)) as c:
            try:
                async with c.stream("POST", endpoint, json=self._body(method, params), headers=headers) as r:
                    if r.status_code != 200:
                        raise A2AError(f"stream request answered {r.status_code}")
                    buf: List[str] = []
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            buf.append(line[5:].lstrip())
                        elif line == "" and buf:
                            try:
                                msg = json.loads("\n".join(buf))
                            except json.JSONDecodeError:
                                raise A2AError("stream event is not valid JSON")
                            buf = []
                            if isinstance(msg, dict) and "error" in msg:
                                err = msg["error"] if isinstance(msg["error"], dict) else {}
                                raise A2AError(str(err.get("message", "agent error")), err.get("code"))
                            res = msg.get("result") if isinstance(msg, dict) else None
                            if isinstance(res, dict):
                                yield res
            except httpx.RequestError as e:
                raise A2AError(f"stream failed: {e}")

    # --- operations -------------------------------------------------------------
    def message_params(self, text: str, context_id: Optional[str] = None, task_id: Optional[str] = None):
        msg: Dict[str, Any] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
        if context_id:
            msg["contextId"] = context_id
        if task_id:
            msg["taskId"] = task_id
        return {"message": msg}

    async def send_message(self, text, context_id=None, task_id=None):
        return await self.call("SendMessage", self.message_params(text, context_id, task_id))

    def send_streaming_message(self, text, context_id=None, task_id=None):
        return self.stream("SendStreamingMessage", self.message_params(text, context_id, task_id))

    async def get_task(self, task_id: str, history_length: Optional[int] = None):
        p: Dict[str, Any] = {"id": task_id}
        if history_length is not None:
            p["historyLength"] = history_length
        return await self.call("GetTask", p)

    async def cancel_task(self, task_id: str):
        return await self.call("CancelTask", {"id": task_id})


# --- rendering: every agent-supplied value is wrapped or validated -------------------
def safe_id(v) -> Optional[str]:
    return v if isinstance(v, str) and _ID.match(v) else None


def safe_enum(v) -> Optional[str]:
    return v if isinstance(v, str) and _ENUM.match(v) else None


def render_part(part, src: str) -> Dict[str, Any]:
    if not isinstance(part, dict):
        return {"type": "unknown"}
    out: Dict[str, Any]
    if "text" in part:
        out = {"type": "text", "text": wrap_untrusted_text(part["text"], src)}
    elif "data" in part:
        out = {"type": "data", "data": wrap_untrusted_data(part["data"], src)}
    elif "url" in part:
        out = {"type": "url", "url": wrap_untrusted_text(part["url"], src)}
    elif "raw" in part:
        out = {"type": "raw", "omitted_base64_chars": len(str(part["raw"]))}
    else:
        return {"type": "unknown"}
    for k in ("mediaType", "filename"):
        if k in part:
            out[k] = wrap_untrusted_text(part[k], src)
    return out


def render_parts(parts, src: str) -> List[Dict[str, Any]]:
    return [render_part(p, src) for p in parts] if isinstance(parts, list) else []


def render_message(msg, src: str) -> Dict[str, Any]:
    msg = msg if isinstance(msg, dict) else {}
    return {
        "role": safe_enum(msg.get("role")),
        "message_id": safe_id(msg.get("messageId")),
        "context_id": safe_id(msg.get("contextId")),
        "task_id": safe_id(msg.get("taskId")),
        "parts": render_parts(msg.get("parts"), src),
    }


def render_artifact(a, src: str) -> Dict[str, Any]:
    a = a if isinstance(a, dict) else {}
    out = {"artifact_id": safe_id(a.get("artifactId")), "parts": render_parts(a.get("parts"), src)}
    if "name" in a:
        out["name"] = wrap_untrusted_text(a["name"], src)
    return out


def render_status(status, src: str) -> Dict[str, Any]:
    status = status if isinstance(status, dict) else {}
    out: Dict[str, Any] = {"state": safe_enum(status.get("state"))}
    if status.get("message"):
        out["message"] = render_message(status["message"], src)
    return out


def render_task(task, src: str, history: bool = True) -> Dict[str, Any]:
    task = task if isinstance(task, dict) else {}
    out: Dict[str, Any] = {
        "task_id": safe_id(task.get("id")),
        "context_id": safe_id(task.get("contextId")),
        **render_status(task.get("status"), src),
    }
    if isinstance(task.get("artifacts"), list):
        out["artifacts"] = [render_artifact(a, src) for a in task["artifacts"]]
    if history and isinstance(task.get("history"), list):
        out["history"] = [render_message(m, src) for m in task["history"]]
    return out


def render_send_result(result: Dict[str, Any], src: str) -> Dict[str, Any]:
    """SendMessage returns exactly one of `task` or `message`."""
    if "task" in result:
        return {"kind": "task", **render_task(result["task"], src)}
    if "message" in result:
        return {"kind": "message", **render_message(result["message"], src)}
    raise A2AError("SendMessage result holds neither task nor message")


def render_stream_event(ev: Dict[str, Any], src: str) -> Dict[str, Any]:
    if "task" in ev:
        return {"event": "task", **render_task(ev["task"], src)}
    if "message" in ev:
        return {"event": "message", **render_message(ev["message"], src)}
    if "statusUpdate" in ev:
        u = ev["statusUpdate"] if isinstance(ev["statusUpdate"], dict) else {}
        return {"event": "status", "task_id": safe_id(u.get("taskId")),
                "context_id": safe_id(u.get("contextId")), **render_status(u.get("status"), src)}
    if "artifactUpdate" in ev:
        u = ev["artifactUpdate"] if isinstance(ev["artifactUpdate"], dict) else {}
        return {"event": "artifact", "task_id": safe_id(u.get("taskId")),
                "context_id": safe_id(u.get("contextId")),
                "append": bool(u.get("append")), "last_chunk": bool(u.get("lastChunk")),
                "artifact": render_artifact(u.get("artifact"), src)}
    return {"event": "unknown"}


def error_text(e: Exception, src: str) -> str:
    """An exception message may carry agent-supplied text."""
    return wrap_untrusted_text(str(e), src)


def make_client(url: str) -> A2AV1Client:
    return A2AV1Client(url)
