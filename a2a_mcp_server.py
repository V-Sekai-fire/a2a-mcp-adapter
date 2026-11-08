"""
Enhanced MCP A2A Bridge with improved task management and result retrieval.

This script implements a bridge between the MCP protocol and A2A protocol,
allowing MCP clients to interact with A2A agents.

Supports multiple transport types:
- stdio: Standard input/output transport
- streamable-http: Recommended HTTP transport
- sse: Server-Sent Events transport

Configure the transport type using the MCP_TRANSPORT environment variable.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional
import logging
import atexit

# Set required environment variable for FastMCP 2.8.1+
os.environ.setdefault('FASTMCP_LOG_LEVEL', 'INFO')
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from common.server.server import A2AServer
from common.server.task_manager import BridgeTaskManager
from persistence_utils import save_to_json, load_from_json
from a2a_v1 import (
    A2AError,
    error_text,
    fetch_card,
    make_client,
    pick_interface,
    render_send_result,
    render_stream_event,
    render_task,
    MAX_STREAM_EVENTS,
)
from security import (
    BearerAuthMiddleware,
    UNTRUSTED_NOTICE,
    require_auth_for_host,
    validate_agent_url,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a FastMCP server
mcp = FastMCP("A2A Bridge Server")

# File paths for persistent storage
DATA_DIR = os.environ.get("A2A_MCP_DATA_DIR", ".")
REGISTERED_AGENTS_FILE = os.path.join(DATA_DIR, "registered_agents.json")
TASK_AGENT_MAPPING_FILE = os.path.join(DATA_DIR, "task_agent_mapping.json")

MAX_PERSISTED_AGENTS = 1000
# Load stored data from disk if it exists
stored_agents = load_from_json(REGISTERED_AGENTS_FILE)
stored_tasks = load_from_json(TASK_AGENT_MAPPING_FILE)

# Initialize in-memory dictionaries with stored data
registered_agents = {}
task_agent_mapping = {}

# Register function to save data on exit
def save_data_on_exit():
    logger.info("Saving data before exit...")
    # Convert registered_agents to a serializable format
    agents_data = {url: agent.model_dump() for url, agent in registered_agents.items()}
    save_to_json(agents_data, REGISTERED_AGENTS_FILE)
    save_to_json(task_agent_mapping, TASK_AGENT_MAPPING_FILE)
    logger.info("Data saved successfully")

atexit.register(save_data_on_exit)

# Periodically save data (every 5 minutes)
async def periodic_save():
    while True:
        await asyncio.sleep(300)  # 5 minutes
        logger.info("Performing periodic data save...")
        agents_data = {url: agent.model_dump() for url, agent in registered_agents.items()}
        save_to_json(agents_data, REGISTERED_AGENTS_FILE)
        save_to_json(task_agent_mapping, TASK_AGENT_MAPPING_FILE)
        logger.info("Periodic save completed")

# Load transport configuration from environment variables
DEFAULT_TRANSPORT = "stdio"
TRANSPORT_TYPES = ["stdio", "streamable-http", "sse"]

# MCP server configuration
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", DEFAULT_TRANSPORT).lower()
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")  # For streamable-http
MCP_SSE_PATH = os.environ.get("MCP_SSE_PATH", "/sse")  # For sse

# A2A server configuration
A2A_HOST = os.environ.get("A2A_HOST", "127.0.0.1")
A2A_PORT = int(os.environ.get("A2A_PORT", "41241"))

# Validate transport type
if MCP_TRANSPORT not in TRANSPORT_TYPES:
    print(f"Warning: Invalid transport type '{MCP_TRANSPORT}'. Using default: {DEFAULT_TRANSPORT}")
    MCP_TRANSPORT = DEFAULT_TRANSPORT

class AgentInfo(BaseModel):
    """Information about an A2A agent."""
    url: str = Field(description="URL of the A2A agent")
    name: str = Field(description="Name of the A2A agent")
    description: str = Field(description="Description of the A2A agent")

async def fetch_agent_card(url: str) -> Dict[str, Any]:
    """Fetch and validate a v1.0 agent card (SSRF-guarded); raises ValueError."""
    try:
        card = await fetch_card(url)
        pick_interface(card)
        return card
    except A2AError as e:
        raise ValueError(str(e))


@mcp.tool()
async def register_agent(url: str, ctx: Context) -> Dict[str, Any]:
    """
    Register an A2A agent with the bridge server.
    
    Args:
        url: URL of the A2A agent
        
    Returns:
        Dictionary with registration status
    """
    try:
        # Fetch the agent card directly
        agent_card = await fetch_agent_card(url)
        
        # Store the agent information
        agent_info = AgentInfo(
            url=url,
            name=str(agent_card.get("name"))[:200],
            description=str(agent_card.get("description") or "No description provided")[:2000],
        )
        
        registered_agents[url] = agent_info
        
        # Save to disk immediately
        agents_data = {url: agent.model_dump() for url, agent in registered_agents.items()}
        save_to_json(agents_data, REGISTERED_AGENTS_FILE)
        
        await ctx.info(f"Successfully registered agent: {agent_info.name}")
        return {
            "status": "success",
            "agent": agent_info.model_dump(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to register agent: {str(e)}",
        }


@mcp.tool()
async def list_agents() -> List[Dict[str, Any]]:
    """
    List all registered A2A agents.
    
    Returns:
        List of registered agents
    """
    return [agent.model_dump() for agent in registered_agents.values()]


@mcp.tool()
async def unregister_agent(url: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Unregister an A2A agent from the bridge server.
    
    Args:
        url: URL of the A2A agent to unregister
        
    Returns:
        Dictionary with unregistration status
    """
    if url not in registered_agents:
        return {
            "status": "error",
            "message": f"Agent not registered: {url}",
        }
    
    try:
        # Get agent name before removing it
        agent_name = registered_agents[url].name
        
        # Remove from registered agents
        del registered_agents[url]
        
        # Clean up any task mappings related to this agent
        # Create a list of task_ids to remove to avoid modifying the dictionary during iteration
        tasks_to_remove = []
        for task_id, agent_url in task_agent_mapping.items():
            if agent_url == url:
                tasks_to_remove.append(task_id)
        
        # Now remove the task mappings
        for task_id in tasks_to_remove:
            del task_agent_mapping[task_id]
        
        # Save changes to disk immediately
        agents_data = {url: agent.model_dump() for url, agent in registered_agents.items()}
        save_to_json(agents_data, REGISTERED_AGENTS_FILE)
        save_to_json(task_agent_mapping, TASK_AGENT_MAPPING_FILE)
        
        if ctx:
            await ctx.info(f"Successfully unregistered agent: {agent_name}")
        
        return {
            "status": "success",
            "message": f"Successfully unregistered agent: {agent_name}",
            "removed_tasks": len(tasks_to_remove),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error unregistering agent: {str(e)}",
        }


@mcp.tool(annotations={"openWorldHint": True})
async def send_message(
    agent_url: str,
    message: str,
    session_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Send a message to an A2A (v1.0) agent.

    Args:
        agent_url: URL of the A2A agent
        message: Message to send
        session_id: Optional A2A contextId for multi-turn conversations

    Returns:
        Agent's response (task or message). All agent text is wrapped as untrusted.
    """
    if agent_url not in registered_agents:
        return {"status": "error", "message": f"Agent not registered: {agent_url}"}
    try:
        if ctx:
            await ctx.info("Sending message to agent")
        result = await make_client(agent_url).send_message(message, context_id=session_id)
        rendered = render_send_result(result, agent_url)
        if rendered.get("task_id"):
            task_agent_mapping[rendered["task_id"]] = agent_url
            save_to_json(task_agent_mapping, TASK_AGENT_MAPPING_FILE)
        if ctx:
            await ctx.info("Received response from agent")
        return {"status": "success", "untrusted_content": UNTRUSTED_NOTICE, "session_id": rendered.get("context_id"), **rendered}
    except Exception as e:
        return {"status": "error", "message": f"Error sending message: {error_text(e, agent_url)}"}


@mcp.tool(annotations={"openWorldHint": True})
async def get_task_result(
    task_id: str,
    history_length: Optional[int] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Get the current state of a task started through send_message.

    Args:
        task_id: ID of the task
        history_length: Optional number of recent history messages to include

    Returns:
        Task state, message, artifacts and history. All agent text is wrapped as untrusted.
    """
    if task_id not in task_agent_mapping:
        return {"status": "error", "message": f"Task ID not found: {task_id}"}
    agent_url = task_agent_mapping[task_id]
    try:
        if ctx:
            await ctx.info(f"Retrieving task result for task_id: {task_id}")
        task = await make_client(agent_url).get_task(task_id, history_length)
        if ctx:
            await ctx.info("Received task result from agent")
        rendered = render_task(task, agent_url)
        return {"status": "success", "untrusted_content": UNTRUSTED_NOTICE, "session_id": rendered.get("context_id"), **rendered}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving task result: {error_text(e, agent_url)}"}


@mcp.tool()
async def cancel_task(
    task_id: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Cancel a running task on an A2A agent.

    Args:
        task_id: ID of the task to cancel
    """
    if task_id not in task_agent_mapping:
        return {"status": "error", "message": f"Task ID not found: {task_id}"}
    agent_url = task_agent_mapping[task_id]
    try:
        if ctx:
            await ctx.info(f"Cancelling task: {task_id}")
        task = await make_client(agent_url).cancel_task(task_id)
        return {"status": "success", "task_id": task_id, "state": render_task(task, agent_url, history=False).get("state")}
    except Exception as e:
        return {"status": "error", "task_id": task_id, "message": f"Error cancelling task: {error_text(e, agent_url)}"}


@mcp.tool(annotations={"openWorldHint": True})
async def send_message_stream(
    agent_url: str,
    message: str,
    session_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Send a message to an A2A (v1.0) agent and collect the streamed events.

    Args:
        agent_url: URL of the A2A agent
        message: Message to send
        session_id: Optional A2A contextId for multi-turn conversations

    Returns:
        The collected events. All agent text is wrapped as untrusted.
    """
    if agent_url not in registered_agents:
        return {"status": "error", "message": f"Agent not registered: {agent_url}"}
    events: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {"status": "success", "untrusted_content": UNTRUSTED_NOTICE,
                           "session_id": session_id, "state": None, "events": events}
    try:
        if ctx:
            await ctx.info("Sending message to agent (streaming)")
        async for ev in make_client(agent_url).send_streaming_message(message, context_id=session_id):
            r = render_stream_event(ev, agent_url)
            events.append(r)
            if r.get("task_id") and r["task_id"] not in task_agent_mapping:
                task_agent_mapping[r["task_id"]] = agent_url
                out["task_id"] = r["task_id"]
            if r.get("context_id"):
                out["session_id"] = r["context_id"]
            if r.get("state"):
                out["state"] = r["state"]
            if len(events) >= MAX_STREAM_EVENTS:
                out["truncated"] = True
                break
        save_to_json(task_agent_mapping, TASK_AGENT_MAPPING_FILE)
        return out
    except Exception as e:
        out.update(status="error", message=f"Error sending message (stream): {error_text(e, agent_url)}")
        return out


def build_agent_card(token_required: bool) -> Dict[str, Any]:
    """The bridge's v1.0 agent card."""
    public = os.environ.get("A2A_PUBLIC_URL") or f"http://{A2A_HOST}:{A2A_PORT}/"
    card: Dict[str, Any] = {
        "name": "MCP Bridge Agent",
        "description": "A bridge between MCP and A2A protocols. Send a message with "
                       "metadata.agent_url naming a registered agent to forward it.",
        "version": "0.2.0",
        "supportedInterfaces": [{"url": public, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": True, "pushNotifications": False, "extendedAgentCard": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [{
            "id": "mcp-bridge",
            "name": "MCP Bridge",
            "description": "Forwards messages to registered A2A agents",
            "tags": ["bridge", "proxy", "mcp", "a2a"],
        }],
    }
    if token_required:
        card["securitySchemes"] = {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}}
        card["securityRequirements"] = [{"schemes": {"bearer": {"list": []}}}]
    return card


def a2a_server_enabled() -> bool:
    return os.environ.get("A2A_SERVER", "").lower() in ("1", "true", "yes")


def setup_a2a_server() -> A2AServer:
    """Build the A2A v1.0 server; it only listens when A2A_SERVER=1."""
    token = require_auth_for_host(A2A_HOST, "A2A listener")
    manager = BridgeTaskManager(is_registered=lambda url: url in registered_agents)
    return A2AServer(manager, build_agent_card(bool(token)), host=A2A_HOST, port=A2A_PORT, token=token)


def _auth_middleware():
    """Bearer-auth middleware list for HTTP transports; refuses open non-loopback binds."""
    from starlette.middleware import Middleware

    token = require_auth_for_host(MCP_HOST, "MCP HTTP listener")
    return [Middleware(BearerAuthMiddleware, token=token)] if token else []


async def main_async():
    """
    Main async function to start both the MCP and A2A servers.
    """
    # Load stored data into memory
    load_registered_agents()
    
    # Start periodic save task
    asyncio.create_task(periodic_save())
    
    # Set up and start the A2A server
    if a2a_server_enabled():
        a2a_task = asyncio.create_task(setup_a2a_server().start_async())
    else:
        a2a_task = None
    
    # Start the MCP server with the configured transport
    print(f"Starting MCP server with {MCP_TRANSPORT} transport...")
    
    if MCP_TRANSPORT == "stdio":
        # Use stdio transport (default)
        mcp_task = asyncio.create_task(
            mcp.run_async(transport="stdio")
        )
    elif MCP_TRANSPORT == "streamable-http":
        # Use streamable-http transport
        auth_middleware = _auth_middleware()
        mcp_task = asyncio.create_task(
            mcp.run_async(
                transport="streamable-http",
                host=MCP_HOST,
                port=MCP_PORT,
                path=MCP_PATH,
                middleware=auth_middleware,
            )
        )
    elif MCP_TRANSPORT == "sse":
        # Use sse transport (deprecated but still supported)
        auth_middleware = _auth_middleware()
        mcp_task = asyncio.create_task(
            mcp.run_async(
                transport="sse",
                host=MCP_HOST,
                port=MCP_PORT,
                path=MCP_SSE_PATH,
                middleware=auth_middleware,
            )
        )
    
    # Run both servers
    await asyncio.gather(*[t for t in (a2a_task, mcp_task) if t is not None])


def load_registered_agents():
    """Load registered agents from stored data on startup."""
    global registered_agents, task_agent_mapping
    
    logger.info("Loading saved data...")
    
    # Load agents data
    agents_data = load_from_json(REGISTERED_AGENTS_FILE)
    for url, agent_data in list(agents_data.items())[:MAX_PERSISTED_AGENTS]:
        try:
            validate_agent_url(url)
            registered_agents[url] = AgentInfo(**agent_data)
        except Exception as e:  # unsafe URL or malformed record: drop it
            logger.warning(f"Dropping persisted agent {url!r}: {e}")
    
    # Load task mappings
    task_agent_mapping = load_from_json(TASK_AGENT_MAPPING_FILE)
    
    logger.info(f"Loaded {len(registered_agents)} agents and {len(task_agent_mapping)} task mappings")


def main():
    """
    Main entry point.
    """
    # Print the configuration for debugging
    print(f"MCP Bridge Configuration:")
    print(f"- Transport: {MCP_TRANSPORT}")
    print(f"- Host: {MCP_HOST}")
    print(f"- Port: {MCP_PORT}")
    if MCP_TRANSPORT == "streamable-http":
        print(f"- Path: {MCP_PATH}")
    elif MCP_TRANSPORT == "sse":
        print(f"- SSE Path: {MCP_SSE_PATH}")
    print(f"A2A Server Configuration:")
    print(f"- Host: {A2A_HOST}")
    print(f"- Port: {A2A_PORT}")
    print(f"Data Storage:")
    print(f"- Registered Agents: {REGISTERED_AGENTS_FILE}")
    print(f"- Task Agent Mapping: {TASK_AGENT_MAPPING_FILE}")
    
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
