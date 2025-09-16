# Parakeet ACR MCP Server Usage Guide

## Overview

This is a Model Context Protocol (MCP) server implementation that provides a simple dummy tool returning the value 42. It uses HTTP streamable transport for communication with MCP clients.

## Installation

1. Navigate to the project directory:
   ```bash
   cd parakeet_acr_mcp
   ```

2. Install dependencies:
   ```bash
   uv install
   ```

## Running the Server

### Option 1: Using the package script
```bash
uv run parakeet-acr-mcp
```

### Option 2: Using the convenience script
```bash
uv run python run_server.py
```

### Custom host and port
```bash
uv run parakeet-acr-mcp --host 0.0.0.0 --port 8080
```

## Available Tools

### `get_answer()`
- **Description**: A dummy tool that returns the answer to everything
- **Parameters**: None
- **Returns**: `42` (integer)
- **Usage**: This tool demonstrates the basic MCP tool functionality

## MCP Client Connection

The server exposes its MCP endpoint at: `http://localhost:8000/mcp`

### Example Client Code

See `examples/client_example.py` for a complete example of how to connect to the server:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8000/mcp") as (read_stream, write_stream, _):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("get_answer", {})
```

## Validation

Run the validation script to ensure everything is working correctly:

```bash
uv run python validate_server.py
```

## Architecture

- **Transport**: HTTP Streamable (supports both Server-Sent Events and JSON responses)
- **Framework**: FastMCP (built on FastAPI and Starlette)
- **Python Version**: Requires Python ≥ 3.10
- **Dependencies**: mcp, fastapi, uvicorn

## Development

The server is built using the FastMCP framework, which provides:
- Automatic tool registration via decorators
- HTTP streamable transport support
- Built-in error handling
- Easy integration with existing ASGI applications
