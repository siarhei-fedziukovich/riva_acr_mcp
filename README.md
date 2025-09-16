# Parakeet ACR MCP Server

A Model Context Protocol (MCP) server that provides a dummy tool returning 42.

## Installation

```bash
uv install
```

## Usage

Start the server with HTTP streamable transport:

```bash
uv run parakeet-acr-mcp
```

The server will start on `http://localhost:8000` by default.

## Available Tools

- `get_answer`: A dummy tool that returns the answer to everything (42)

## MCP Integration

This server uses the streamable HTTP transport and can be connected to any MCP-compatible client.

Example client connection URL: `http://localhost:8000/mcp`
