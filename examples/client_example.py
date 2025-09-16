#!/usr/bin/env python3
"""
Example client for connecting to the Parakeet ACR MCP Server.

This demonstrates how to connect to the server using the streamable HTTP transport
and call the available tools.
"""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    """Main client function to demonstrate MCP server interaction."""
    server_url = "http://localhost:8000/mcp"
    
    print(f"Connecting to MCP server at: {server_url}")
    
    try:
        # Connect to the streamable HTTP server
        async with streamablehttp_client(server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            # Create a session using the client streams
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                print("Initializing connection...")
                await session.initialize()
                
                # List available tools
                print("Listing available tools...")
                tools = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Call the get_answer tool
                print("Calling get_answer tool...")
                result = await session.call_tool("get_answer", {})
                print(f"Tool result: {result}")
                
                print("Client example completed successfully!")
                
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running with: uv run parakeet-acr-mcp")


if __name__ == "__main__":
    asyncio.run(main())
