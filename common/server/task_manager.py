"""Bridge handler for the A2A v1.0 server: forwards to registered A2A agents.

The bridge mints its own task ids and maps them to (agent url, remote task id), so a caller
never supplies or learns a remote id. Every message, task and event that crosses the bridge
is rebuilt from a whitelist of fields with size limits. Agent text is passed through
unchanged (the consumer is an A2A client, not an LLM context); the untrusted-reply envelope
belongs to the MCP tool side. Access is scoped to the one bearer-token principal.
"""
import contextlib
import json
import logging
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import a2a_v1
from a2a_v1 import A2AError, safe_id
from common.server.server import RpcError

logger = logging.getLogger(__name__)

TASK_STATES = {
    "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED", "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED", "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_REJECTED", "TASK_STATE_AUTH_REQUIRED",
}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
MAX_TEXT = 100_000
MAX_DATA = 200_000
MAX_RAW = 4_000_000
MAX_PARTS = 64
MAX_ARTIFACTS = 64
MAX_HISTORY = 100
MAX_TASKS = 1000
MAX_EVENTS = 1000
UPSTREAM_CODES = {-32001: "Task not found", -32002: "Task cannot be canceled", -32004: "Operation not supported"}


def _bad(inbound: bool) -> RpcError:
    return RpcError(-32602, "Invalid parameters") if inbound else RpcError(-32006, "Invalid agent response")


def clean_part(p, inbound: bool) -> Dict[str, Any]:
    if not isinstance(p, dict):
        raise _bad(inbound)
    keys = [k for k in ("text", "url", "data", "raw") if k in p]
    if len(keys) != 1:
        raise _bad(inbound)
    k = keys[0]
    v = p[k]
    if k == "data":
        try:
            too_big = len(json.dumps(v)) > MAX_DATA
        except (TypeError, ValueError):
            raise _bad(inbound)
        if too_big:
            raise _bad(inbound)
    else:
        if not isinstance(v, str) or len(v) > (MAX_RAW if k == "raw" else MAX_TEXT):
            raise _bad(inbound)
    out: Dict[str, Any] = {k: v}
    for m in ("mediaType", "filename"):
        if isinstance(p.get(m), str) and len(p[m]) <= 256:
            out[m] = p[m]
    return out


def _parts(parts, inbound: bool) -> List[Dict[str, Any]]:
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
        raise _bad(inbound)
    return [clean_part(p, inbound) for p in parts]


def clean_message(m, inbound: bool, task_id: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(m, dict):
        raise _bad(inbound)
    role = m.get("role")
    if role not in (("ROLE_USER",) if inbound else ("ROLE_AGENT", "ROLE_USER")):
        raise _bad(inbound)
    out: Dict[str, Any] = {
        "messageId": safe_id(m.get("messageId")) or str(uuid.uuid4()),
        "role": role,
        "parts": _parts(m.get("parts"), inbound),
    }
    ctx = m.get("contextId")
    if ctx is not None:
        if not safe_id(ctx):
            raise _bad(inbound)
        out["contextId"] = ctx
    if task_id:
        out["taskId"] = task_id
    return out


def clean_status(s, inbound: bool, task_id: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(s, dict) or s.get("state") not in TASK_STATES:
        raise _bad(inbound)
    out: Dict[str, Any] = {"state": s["state"]}
    if s.get("message") is not None:
        out["message"] = clean_message(s["message"], inbound, task_id)
    if isinstance(s.get("timestamp"), str) and len(s["timestamp"]) <= 64:
        out["timestamp"] = s["timestamp"]
    return out


def clean_artifact(a, inbound: bool = False) -> Dict[str, Any]:
    if not isinstance(a, dict):
        raise _bad(inbound)
    out: Dict[str, Any] = {"artifactId": safe_id(a.get("artifactId")) or str(uuid.uuid4()),
                           "parts": _parts(a.get("parts"), inbound)}
    for k in ("name", "description"):
        if isinstance(a.get(k), str) and len(a[k]) <= 2000:
            out[k] = a[k]
    return out


def clean_task(t, bridge_id: str) -> Dict[str, Any]:
    if not isinstance(t, dict):
        raise _bad(False)
    out: Dict[str, Any] = {"id": bridge_id, "status": clean_status(t.get("status"), False, bridge_id)}
    if t.get("contextId") is not None:
        if not safe_id(t["contextId"]):
            raise _bad(False)
        out["contextId"] = t["contextId"]
    if t.get("artifacts") is not None:
        if not isinstance(t["artifacts"], list) or len(t["artifacts"]) > MAX_ARTIFACTS:
            raise _bad(False)
        out["artifacts"] = [clean_artifact(a) for a in t["artifacts"]]
    if t.get("history") is not None:
        if not isinstance(t["history"], list) or len(t["history"]) > MAX_HISTORY:
            raise _bad(False)
        out["history"] = [clean_message(m, False, bridge_id) for m in t["history"]]
    return out


def _opt_int(params: Dict[str, Any], key: str, lo: int, hi: int) -> Optional[int]:
    v = params.get(key)
    if v is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise RpcError(-32602, "Invalid parameters")
    return v


class BridgeTaskManager:
    def __init__(self, is_registered: Callable[[str], bool]):
        self.is_registered = is_registered
        self._clients: Dict[str, Any] = {}
        self._tasks: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._reverse: Dict[Any, str] = {}

    # --- bookkeeping --------------------------------------------------------------
    def _client(self, url: str):
        if url not in self._clients:
            self._clients[url] = a2a_v1.make_client(url)
        return self._clients[url]

    def _bind(self, url: str, remote_id) -> str:
        if not safe_id(remote_id):
            raise _bad(False)
        key = (url, remote_id)
        if key in self._reverse:
            self._tasks.move_to_end(self._reverse[key])
            return self._reverse[key]
        bid = str(uuid.uuid4())
        self._tasks[bid] = {"url": url, "remote_id": remote_id, "snapshot": None}
        self._reverse[key] = bid
        while len(self._tasks) > MAX_TASKS:
            _, old = self._tasks.popitem(last=False)
            self._reverse.pop((old["url"], old["remote_id"]), None)
        return bid

    def _entry(self, bridge_id) -> Dict[str, Any]:
        entry = self._tasks.get(bridge_id) if isinstance(bridge_id, str) else None
        if entry is None or not self.is_registered(entry["url"]):
            raise RpcError(-32001, "Task not found")
        self._tasks.move_to_end(bridge_id)
        return entry

    def _upstream(self, e: A2AError) -> RpcError:
        if e.code in UPSTREAM_CODES:
            return RpcError(e.code, UPSTREAM_CODES[e.code])
        logger.warning("upstream agent error: %r", str(e)[:200])
        return RpcError(-32006, "Upstream agent failed or returned an invalid response")

    def _prepare(self, params):
        """Validate a send request; return (agent url, remote message)."""
        if not isinstance(params, dict) or not isinstance(params.get("message"), dict):
            raise RpcError(-32602, "Invalid parameters")
        raw = params["message"]
        remote_task = None
        if raw.get("taskId") is not None:
            entry = self._entry(raw["taskId"])
            url, remote_task = entry["url"], entry["remote_id"]
        else:
            md = params.get("metadata")
            url = md.get("agent_url") if isinstance(md, dict) else None
            if not isinstance(url, str) or not self.is_registered(url):
                raise RpcError(-32602, "metadata.agent_url must name a registered agent")
        return url, clean_message(raw, True, remote_task)

    def _remember(self, bid: str, task: Dict[str, Any]) -> None:
        self._tasks[bid]["snapshot"] = task

    # --- unary methods ------------------------------------------------------------
    async def send_message(self, params) -> Dict[str, Any]:
        url, msg = self._prepare(params)
        try:
            result = await self._client(url).call("SendMessage", {"message": msg})
        except A2AError as e:
            raise self._upstream(e)
        if isinstance(result.get("task"), dict):
            bid = self._bind(url, result["task"].get("id"))
            task = clean_task(result["task"], bid)
            self._remember(bid, task)
            return {"task": task}
        if isinstance(result.get("message"), dict):
            return {"message": clean_message(result["message"], False)}
        raise _bad(False)

    async def get_task(self, params) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise RpcError(-32602, "Invalid parameters")
        entry = self._entry(params.get("id"))
        hl = _opt_int(params, "historyLength", 0, MAX_HISTORY)
        try:
            remote = await self._client(entry["url"]).get_task(entry["remote_id"], hl)
        except A2AError as e:
            raise self._upstream(e)
        task = clean_task(remote, params["id"])
        self._remember(params["id"], task)
        return task

    async def cancel_task(self, params) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise RpcError(-32602, "Invalid parameters")
        entry = self._entry(params.get("id"))
        try:
            remote = await self._client(entry["url"]).cancel_task(entry["remote_id"])
        except A2AError as e:
            raise self._upstream(e)
        task = clean_task(remote, params["id"])
        self._remember(params["id"], task)
        return task

    async def list_tasks(self, params) -> Dict[str, Any]:
        params = {} if params is None else params
        if not isinstance(params, dict):
            raise RpcError(-32602, "Invalid parameters")
        status, ctx = params.get("status"), params.get("contextId")
        if status is not None and status not in TASK_STATES:
            raise RpcError(-32602, "Invalid parameters")
        if ctx is not None and not safe_id(ctx):
            raise RpcError(-32602, "Invalid parameters")
        size = _opt_int(params, "pageSize", 1, 100) or 50
        hl = _opt_int(params, "historyLength", 0, MAX_HISTORY)
        token = params.get("pageToken") or "0"
        if not isinstance(token, str) or not token.isdigit() or len(token) > 9:
            raise RpcError(-32602, "Invalid parameters")
        offset = int(token)
        matches = []
        for entry in reversed(self._tasks.values()):
            snap = entry["snapshot"]
            if snap is None or not self.is_registered(entry["url"]):
                continue
            if status and snap["status"]["state"] != status:
                continue
            if ctx and snap.get("contextId") != ctx:
                continue
            matches.append(snap)
        page = []
        for snap in matches[offset:offset + size]:
            snap = dict(snap)
            if "history" in snap:
                snap["history"] = snap["history"][-hl:] if hl else []
            page.append(snap)
        more = offset + size < len(matches)
        return {"tasks": page, "nextPageToken": str(offset + size) if more else "",
                "pageSize": size, "totalSize": len(matches)}

    # --- streaming methods --------------------------------------------------------
    def _map_event(self, ev: Dict[str, Any], url: str) -> Dict[str, Any]:
        if isinstance(ev.get("task"), dict):
            bid = self._bind(url, ev["task"].get("id"))
            task = clean_task(ev["task"], bid)
            self._remember(bid, task)
            return {"task": task}
        if isinstance(ev.get("message"), dict):
            return {"message": clean_message(ev["message"], False)}
        for key in ("statusUpdate", "artifactUpdate"):
            u = ev.get(key)
            if not isinstance(u, dict):
                continue
            bid = self._bind(url, u.get("taskId"))
            if not safe_id(u.get("contextId")):
                raise _bad(False)
            out: Dict[str, Any] = {"taskId": bid, "contextId": u["contextId"]}
            if key == "statusUpdate":
                out["status"] = clean_status(u.get("status"), False, bid)
                snap = self._tasks[bid]["snapshot"]
                if snap is not None:
                    snap["status"] = out["status"]
            else:
                out["artifact"] = clean_artifact(u.get("artifact"))
                out["append"] = bool(u.get("append"))
                out["lastChunk"] = bool(u.get("lastChunk"))
            return {key: out}
        raise _bad(False)

    async def _relay(self, url: str, method: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        count = 0
        try:
            async with contextlib.aclosing(self._client(url).stream(method, params)) as stream:
                async for ev in stream:
                    yield self._map_event(ev, url)
                    count += 1
                    if count >= MAX_EVENTS:
                        break
        except A2AError as e:
            raise self._upstream(e)

    async def send_streaming_message(self, params) -> AsyncIterator[Dict[str, Any]]:
        url, msg = self._prepare(params)
        async for ev in self._relay(url, "SendStreamingMessage", {"message": msg}):
            yield ev

    async def subscribe_to_task(self, params) -> AsyncIterator[Dict[str, Any]]:
        if not isinstance(params, dict):
            raise RpcError(-32602, "Invalid parameters")
        entry = self._entry(params.get("id"))
        snap = entry["snapshot"]
        if snap is not None and snap["status"]["state"] in TERMINAL:
            raise RpcError(-32004, "Task is in a terminal state")
        async for ev in self._relay(entry["url"], "SubscribeToTask", {"id": entry["remote_id"]}):
            yield ev
