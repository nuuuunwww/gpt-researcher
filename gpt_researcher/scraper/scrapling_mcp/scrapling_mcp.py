import asyncio
import json
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup

from ..utils import get_relevant_images


class ScraplingMCPScraper:
    """
    Scraper backend that delegates page extraction to a Scrapling MCP server.

    Environment variables:
      SCRAPLING_MCP_URL: Scrapling MCP base URL. Default: http://scrapling:8000
      SCRAPLING_MCP_TOOL: MCP tool name. Default: get
      SCRAPLING_MCP_TIMEOUT: HTTP timeout in seconds. Default: 60
      SCRAPLING_MCP_PROXY: Optional proxy passed to Scrapling.
      SCRAPLING_MCP_EXTRA_ARGS_JSON: Optional JSON object merged into tool arguments.
    """

    def __init__(self, link: str, session=None):
        self.link = link
        self.session = session
        self.base_url = os.getenv("SCRAPLING_MCP_URL", "http://scrapling:8000").rstrip("/")
        self.tool_name = os.getenv("SCRAPLING_MCP_TOOL", "get")
        self.timeout = self._load_timeout()
        self.proxy = os.getenv("SCRAPLING_MCP_PROXY")
        self.extra_args = self._load_extra_args()

    def scrape(self) -> tuple[str, list, str]:
        return asyncio.run(self.scrape_async())

    async def scrape_async(self) -> tuple[str, list, str]:
        try:
            result = await self._call_scrapling_mcp()
            content = self._extract_content(result)
            title = self._extract_title(result, content)
            image_urls = self._extract_images(content)
            return content, image_urls, title
        except Exception as exc:
            print(f"Scrapling MCP error for {self.link}: {exc}")
            return "", [], ""

    def _load_extra_args(self) -> dict[str, Any]:
        raw = os.getenv("SCRAPLING_MCP_EXTRA_ARGS_JSON", "{}")
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            print(f"Invalid SCRAPLING_MCP_EXTRA_ARGS_JSON: {exc}")
            return {}

        if isinstance(parsed, dict):
            return parsed

        print("Invalid SCRAPLING_MCP_EXTRA_ARGS_JSON: value must be a JSON object")
        return {}

    def _load_timeout(self) -> float:
        raw = os.getenv("SCRAPLING_MCP_TIMEOUT", "60")
        try:
            return float(raw)
        except ValueError:
            print(f"Invalid SCRAPLING_MCP_TIMEOUT: {raw}")
            return 60.0

    async def _call_scrapling_mcp(self) -> dict[str, Any]:
        endpoint = f"{self.base_url}/mcp"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        arguments: dict[str, Any] = {
            "url": self.link,
            "extraction_type": "markdown",
            "main_content_only": True,
        }

        if self.proxy:
            arguments["proxy"] = self.proxy

        arguments.update(self.extra_args)

        initialize_payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "gpt-researcher",
                    "version": "0.14.7",
                },
            },
        }

        initialized_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }

        tool_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": self.tool_name,
                "arguments": arguments,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                endpoint, json=initialize_payload, headers=headers
            )
            response.raise_for_status()
            data = self._decode_mcp_response(response)

            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(data["error"])

            session_id = response.headers.get("mcp-session-id")
            session_headers = headers.copy()
            if session_id:
                session_headers["mcp-session-id"] = session_id

            response = await client.post(
                endpoint, json=initialized_payload, headers=session_headers
            )
            response.raise_for_status()

            response = await client.post(
                endpoint, json=tool_payload, headers=session_headers
            )
            response.raise_for_status()
            data = self._decode_mcp_response(response)

        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])

        if isinstance(data, dict):
            return data.get("result", data)

        raise RuntimeError("Unexpected Scrapling MCP response shape")

    def _decode_mcp_response(self, response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            data = response.json()
            if isinstance(data, dict):
                return data
            raise RuntimeError("Unexpected Scrapling MCP response shape")

        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue

            data = json.loads(line.removeprefix("data:").strip())
            if isinstance(data, dict):
                return data

        raise RuntimeError("Unexpected Scrapling MCP event stream response")

    def _extract_content(self, data: dict[str, Any]) -> str:
        candidates: list[Any] = []

        if isinstance(data, dict):
            for structured_key in ("structuredContent", "structured_content"):
                structured = data.get(structured_key)
                if isinstance(structured, dict):
                    candidates.extend(
                        [
                            structured.get("content"),
                            structured.get("markdown"),
                            structured.get("text"),
                            structured.get("html"),
                            structured.get("body"),
                        ]
                    )

            candidates.extend(
                [
                    data.get("content"),
                    data.get("markdown"),
                    data.get("text"),
                    data.get("html"),
                    data.get("body"),
                ]
            )

            content = data.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        candidates.extend(
                            [
                                item.get("text"),
                                item.get("content"),
                                item.get("markdown"),
                                item.get("html"),
                            ]
                        )

        for candidate in candidates:
            content = self._coerce_content(candidate)
            if content:
                return content

        return ""

    def _coerce_content(self, candidate: Any) -> str:
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return ""

            try:
                parsed = json.loads(text)
            except ValueError:
                return text

            if isinstance(parsed, dict):
                return self._extract_content(parsed)

            return text

        if isinstance(candidate, list):
            parts = []
            for item in candidate:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            return "\n\n".join(parts)

        return ""

    def _extract_title(self, data: dict[str, Any], content: str) -> str:
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict):
            title = metadata.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()

        if isinstance(data, dict):
            title = data.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()

        if content:
            return content.strip().splitlines()[0].strip()[:120]

        return ""

    def _extract_images(self, content: str) -> list:
        if not content:
            return []

        lower_content = content.lower()
        if "<html" not in lower_content and "<img" not in lower_content:
            return []

        try:
            soup = BeautifulSoup(content, "lxml")
            return get_relevant_images(soup, self.link)
        except Exception:
            return []
