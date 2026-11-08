"""A2A v1.0 server (JSON-RPC binding), https://a2a-protocol.org/latest/specification/ (1.0.0).

Methods: SendMessage, SendStreamingMessage, GetTask, ListTasks, CancelTask, SubscribeToTask.
Push-notification config methods and GetExtendedAgentCard answer the spec's
"not supported" errors. The agent card at /.well-known/agent-card.json is public; the
JSON-RPC endpoint needs the bearer token when one is configured.
"""
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from security import BearerAuthMiddleware

logger = logging.getLogger(__name__)

CARD_PATH = "/.well-known/agent-card.json"
SUPPORTED_MAJOR = "1"
MAX_BODY = 8 * 1024 * 1024

UNARY = {
    "SendMessage": "send_message",
    "GetTask": "get_task",
    "ListTasks": "list_tasks",
    "CancelTask": "cancel_task",
}
STREAMING = {
    "SendStreamingMessage": "send_streaming_message",
    "SubscribeToTask": "subscribe_to_task",
}
PUSH_CONFIG = {
    "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig",
    "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig",
}


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error_body(rid, e: RpcError) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": e.code, "message": e.message}}


async def _read_body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY:
        raise RpcError(-32600, "Request too large")
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY:
            raise RpcError(-32600, "Request too large")
    return bytes(buf)


def _check_version(request: Request) -> None:
    # A missing header means a 0.3 client (spec 3.6.1); this server speaks 1.x only.
    version = request.headers.get("a2a-version", "").strip()
    if version.split(".")[0] != SUPPORTED_MAJOR:
        raise RpcError(-32009, "Unsupported A2A version; this agent supports 1.0")


class A2AServer:
    def __init__(self, handler, card: Dict[str, Any], host: str = "127.0.0.1",
                 port: int = 41241, token: Optional[str] = None):
        self.handler = handler
        self.card = card
        self.host = host
        self.port = port
        inner = Starlette(routes=[
            Route(CARD_PATH, self._card, methods=["GET"]),
            Route("/", self._rpc, methods=["POST"]),
        ])
        # The card is public so clients can discover the bearer scheme.
        self.app = BearerAuthMiddleware(inner, token=token, exempt_paths=(CARD_PATH,))

    async def start_async(self) -> None:
        import uvicorn
        await uvicorn.Server(uvicorn.Config(self.app, host=self.host, port=self.port,
                                            log_level="info")).serve()

    async def _card(self, request: Request) -> JSONResponse:
        return JSONResponse(self.card)

    async def _rpc(self, request: Request):
        rid = None
        try:
            raw = await _read_body(request)
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise RpcError(-32700, "Invalid JSON payload")
            if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
                raise RpcError(-32600, "Request payload validation error")
            candidate = body.get("id")
            if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
                rid = candidate
            _check_version(request)
            method = body["method"]
            params = body.get("params", {})
            if method in UNARY:
                result = await getattr(self.handler, UNARY[method])(params)
                return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})
            if method in STREAMING:
                return await self._stream(rid, getattr(self.handler, STREAMING[method])(params))
            if method in PUSH_CONFIG:
                raise RpcError(-32003, "Push notifications are not supported")
            if method == "GetExtendedAgentCard":
                raise RpcError(-32004, "Extended agent card is not supported")
            raise RpcError(-32601, "Method not found")
        except RpcError as e:
            return JSONResponse(_error_body(rid, e))
        except Exception:
            logger.exception("unhandled error in JSON-RPC handler")
            return JSONResponse(_error_body(rid, RpcError(-32603, "Internal error")))

    async def _stream(self, rid, events: AsyncIterator[Dict[str, Any]]):
        it = events.__aiter__()
        # Pull the first event now so setup errors become a plain JSON-RPC error.
        try:
            first = await it.__anext__()
        except StopAsyncIteration:
            first = None

        def frame(payload: Dict[str, Any]) -> str:
            return "data: " + json.dumps(payload) + "\n\n"

        async def body():
            try:
                if first is not None:
                    yield frame({"jsonrpc": "2.0", "id": rid, "result": first})
                async for ev in it:
                    yield frame({"jsonrpc": "2.0", "id": rid, "result": ev})
            except RpcError as e:
                yield frame(_error_body(rid, e))
            except Exception:
                logger.exception("unhandled error in stream")
                yield frame(_error_body(rid, RpcError(-32603, "Internal error")))

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})
