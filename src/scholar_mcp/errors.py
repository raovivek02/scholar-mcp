"""Error types. Messages are written for an LLM reader: say what failed and what to try next.

``ScholarError`` subclasses the SDK's ``ToolError``/``ResourceError`` on purpose: the SDK only
forwards the text of *deliberately raised* errors to the client; any other exception is
reported as a generic "Error executing tool", which leaves the model unable to recover.
"""

try:  # mcp >= 2.0
    from mcp.server.mcpserver.exceptions import ResourceError, ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp.exceptions import ResourceError, ToolError


class ScholarError(ToolError, ResourceError):
    """Base class for all errors surfaced to the MCP client."""


class InvalidInputError(ScholarError):
    pass


class NotFoundError(ScholarError):
    pass


class RateLimitedError(ScholarError):
    pass


class UpstreamError(ScholarError):
    pass
