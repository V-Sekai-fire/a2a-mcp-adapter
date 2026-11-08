import asyncio, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from security import (BearerAuthMiddleware, UnsafeURLError, require_auth_for_host, validate_agent_url)
import persistence_utils as pu

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data", "http://127.0.0.1:8080", "http://localhost/",
    "http://10.0.0.5/", "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/",
    "file:///etc/passwd", "ftp://example.com/", "http://user:pw@8.8.8.8/", "", "http:///x"])
def test_blocked(url):
    with pytest.raises(UnsafeURLError):
        validate_agent_url(url)

def test_public_ip_allowed():
    assert validate_agent_url("https://8.8.8.8/x") == "https://8.8.8.8/x"

def test_private_opt_in(monkeypatch):
    monkeypatch.setenv("A2A_ALLOW_PRIVATE_AGENTS", "1")
    assert validate_agent_url("http://127.0.0.1:9000")

def test_auth_required_non_loopback(monkeypatch):
    monkeypatch.delenv("A2A_MCP_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        require_auth_for_host("0.0.0.0", "x")
    assert require_auth_for_host("127.0.0.1", "x") is None
    monkeypatch.setenv("A2A_MCP_AUTH_TOKEN", "t")
    assert require_auth_for_host("0.0.0.0", "x") == "t"

def _call(token, header):
    sent = []
    async def app(scope, receive, send): sent.append("app")
    async def send(m): sent.append(m.get("status", m["type"]))
    scope = {"type": "http", "headers": [(b"authorization", header.encode())] if header else []}
    asyncio.run(BearerAuthMiddleware(app, token)(scope, None, send))
    return sent

def test_middleware():
    assert _call("s3", "Bearer s3") == ["app"]
    assert _call("s3", "Bearer nope")[0] == 401
    assert _call("s3", "")[0] == 401
    assert _call(None, "") == ["app"]

def test_persistence(tmp_path):
    f = str(tmp_path / "a.json")
    assert pu.save_to_json({"a": 1}, f)
    assert oct(os.stat(f).st_mode & 0o777) == "0o600"
    assert pu.load_from_json(f) == {"a": 1}
    open(f, "w").write("[1]"); assert pu.load_from_json(f) == {}
    open(f, "w").write("x" * (pu.MAX_FILE_BYTES + 1)); assert pu.load_from_json(f) == {}
