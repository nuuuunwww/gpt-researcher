import pytest

from gpt_researcher.scraper import ArxivScraper, PyMuPDFScraper, Scraper
from gpt_researcher.scraper.scrapling_mcp import ScraplingMCPScraper
from gpt_researcher.utils.workers import WorkerPool


class _FakeResponse:
    def __init__(self, payload, headers=None, text=""):
        self.payload = payload
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@pytest.fixture
def scraper():
    return ScraplingMCPScraper("https://example.com")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"content": "Body"}, "Body"),
        ({"markdown": "# Title\nBody"}, "# Title\nBody"),
        ({"text": "Body"}, "Body"),
        ({"structuredContent": {"markdown": "Body"}}, "Body"),
        ({"structured_content": {"content": "Body"}}, "Body"),
        ({"content": [{"type": "text", "text": "Body"}]}, "Body"),
    ],
)
def test_extract_content(scraper, payload, expected):
    assert scraper._extract_content(payload) == expected


def test_extract_title_uses_metadata(scraper):
    data = {"metadata": {"title": "Example Title"}, "content": "Body"}

    assert scraper._extract_title(data, "Body") == "Example Title"


@pytest.mark.asyncio
async def test_scrape_async_returns_empty_tuple_on_mcp_error(monkeypatch, scraper):
    async def raise_error():
        raise RuntimeError({"message": "failed"})

    monkeypatch.setattr(scraper, "_call_scrapling_mcp", raise_error)

    assert await scraper.scrape_async() == ("", [], "")


@pytest.mark.asyncio
async def test_call_scrapling_mcp_raises_jsonrpc_error(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, endpoint, json, headers):
            return _FakeResponse({"error": {"message": "failed"}})

    monkeypatch.setattr(
        "gpt_researcher.scraper.scrapling_mcp.scrapling_mcp.httpx.AsyncClient",
        FakeAsyncClient,
    )

    with pytest.raises(RuntimeError, match="failed"):
        await ScraplingMCPScraper("https://example.com")._call_scrapling_mcp()


@pytest.mark.asyncio
async def test_call_scrapling_mcp_uses_env_configuration(monkeypatch):
    calls = []
    responses = [
        _FakeResponse({"result": {}}, headers={"mcp-session-id": "session-1"}),
        _FakeResponse({}),
        _FakeResponse({"result": {"content": "Body"}}),
    ]

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, endpoint, json, headers):
            calls.append(
                {
                    "endpoint": endpoint,
                    "headers": headers,
                    "json": json,
                    "timeout": self.timeout,
                }
            )
            return responses.pop(0)

    monkeypatch.setenv("SCRAPLING_MCP_URL", "http://scrapling:8000/")
    monkeypatch.setenv("SCRAPLING_MCP_TOOL", "fetch")
    monkeypatch.setenv("SCRAPLING_MCP_TIMEOUT", "30")
    monkeypatch.setenv("SCRAPLING_MCP_PROXY", "http://proxy:3128")
    monkeypatch.setenv(
        "SCRAPLING_MCP_EXTRA_ARGS_JSON",
        '{"extraction_type":"html","wait":1000}',
    )
    monkeypatch.setattr(
        "gpt_researcher.scraper.scrapling_mcp.scrapling_mcp.httpx.AsyncClient",
        FakeAsyncClient,
    )

    result = await ScraplingMCPScraper("https://example.com")._call_scrapling_mcp()

    assert result == {"content": "Body"}
    assert len(calls) == 3
    assert calls[0]["endpoint"] == "http://scrapling:8000/mcp"
    assert calls[0]["timeout"] == 30.0
    assert calls[0]["json"]["method"] == "initialize"
    assert calls[1]["json"]["method"] == "notifications/initialized"
    assert calls[1]["headers"]["mcp-session-id"] == "session-1"
    assert calls[2] == {
        "endpoint": "http://scrapling:8000/mcp",
        "headers": {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": "session-1",
        },
        "timeout": 30.0,
        "json": {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "fetch",
                "arguments": {
                    "url": "https://example.com",
                    "extraction_type": "html",
                    "main_content_only": True,
                    "proxy": "http://proxy:3128",
                    "wait": 1000,
                },
            },
        },
    }


def test_decode_mcp_response_supports_event_stream(scraper):
    response = _FakeResponse(
        {},
        headers={"content-type": "text/event-stream"},
        text='event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":"Body"}}\n',
    )

    assert scraper._decode_mcp_response(response)["result"] == {"content": "Body"}


def test_invalid_timeout_and_extra_args_fall_back(monkeypatch):
    monkeypatch.setenv("SCRAPLING_MCP_TIMEOUT", "not-a-number")
    monkeypatch.setenv("SCRAPLING_MCP_EXTRA_ARGS_JSON", "[]")

    scraper = ScraplingMCPScraper("https://example.com")

    assert scraper.timeout == 60.0
    assert scraper.extra_args == {}


def test_scraper_registration_selects_scrapling_mcp():
    manager = Scraper(
        urls=["https://example.com"],
        user_agent="test-agent",
        scraper="scrapling_mcp",
        worker_pool=WorkerPool(max_workers=1),
    )

    try:
        assert manager.get_scraper("https://example.com") is ScraplingMCPScraper
    finally:
        manager.worker_pool.executor.shutdown()


def test_scraper_registration_preserves_pdf_and_arxiv_overrides():
    manager = Scraper(
        urls=["https://example.com"],
        user_agent="test-agent",
        scraper="scrapling_mcp",
        worker_pool=WorkerPool(max_workers=1),
    )

    try:
        assert manager.get_scraper("https://example.com/paper.pdf") is PyMuPDFScraper
        assert manager.get_scraper("https://arxiv.org/abs/1234.5678") is ArxivScraper
    finally:
        manager.worker_pool.executor.shutdown()
