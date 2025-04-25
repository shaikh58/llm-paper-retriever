from fastmcp import FastMCP
# import torch

mcp = FastMCP("Demo 🚀")

@mcp.tool()
def add(a: str, b: str) -> str:
    """Add two numbers"""
    return int(a) + int(b)

if __name__ == "__main__":
    mcp.run()