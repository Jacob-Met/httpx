import json

import anyio
import pytest

import httpx


@pytest.mark.anyio
async def test_asgi_streaming_yields_before_app_completes():
    next_chunk = anyio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send(
            {"type": "http.response.body", "body": b"first\n", "more_body": True}
        )
        await next_chunk.wait()
        await send({"type": "http.response.body", "body": b"second\n"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as client:
        with anyio.fail_after(0.5):
            async with client.stream("GET", "http://example.org/stream") as response:
                assert response.status_code == 200
                chunks = []
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    if chunk == b"first\n":
                        next_chunk.set()
                assert chunks == [b"first\n", b"second\n"]


@pytest.mark.anyio
async def test_asgi_streaming_close_cancels_infinite_app():
    stopped = anyio.Event()

    async def app(scope, receive, send):
        try:
            await send({"type": "http.response.start", "status": 200})
            await send(
                {"type": "http.response.body", "body": b"first\n", "more_body": True}
            )
            await anyio.sleep_forever()
        finally:
            stopped.set()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as client:
        with anyio.fail_after(0.5):
            async with client.stream("GET", "http://example.org/stream") as response:
                assert response.status_code == 200
            await stopped.wait()


@pytest.mark.anyio
async def test_asgi_streaming_preserves_backpressure():
    sent_second = anyio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"one", "more_body": True})
        await send({"type": "http.response.body", "body": b"two", "more_body": True})
        sent_second.set()
        await send({"type": "http.response.body", "body": b"three"})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        with anyio.fail_after(1):
            async with client.stream("GET", "http://example.org/stream") as response:
                await anyio.sleep(0.01)
                assert not sent_second.is_set()
                assert [chunk async for chunk in response.aiter_raw()] == [
                    b"one",
                    b"two",
                    b"three",
                ]
                assert sent_second.is_set()


@pytest.mark.anyio
async def test_asgi_streaming_app_error_after_body():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send(
            {"type": "http.response.body", "body": b"partial", "more_body": True}
        )
        raise RuntimeError("midstream failure")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        with pytest.raises(RuntimeError, match="midstream failure"):
            await client.get("http://example.org/failure")


@pytest.mark.anyio
async def test_asgi_application_incomplete_response_is_an_error():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        with pytest.raises(AssertionError, match="did not finish response body"):
            await client.get("http://example.org/incomplete")


@pytest.mark.anyio
async def test_asgi_cancel_while_waiting_for_headers():
    stopped = anyio.Event()

    async def app(scope, receive, send):
        try:
            await anyio.sleep_forever()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        with anyio.move_on_after(0.03) as cancelled:
            await client.get("http://example.org/never-starts")
        assert cancelled.cancel_called
        assert stopped.is_set()


async def hello_world(scope, receive, send):
    status = 200
    output = b"Hello, World!"
    headers = [(b"content-type", "text/plain"), (b"content-length", str(len(output)))]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": output})


async def echo_path(scope, receive, send):
    status = 200
    output = json.dumps({"path": scope["path"]}).encode("utf-8")
    headers = [(b"content-type", "text/plain"), (b"content-length", str(len(output)))]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": output})


async def echo_raw_path(scope, receive, send):
    status = 200
    output = json.dumps({"raw_path": scope["raw_path"].decode("ascii")}).encode("utf-8")
    headers = [(b"content-type", "text/plain"), (b"content-length", str(len(output)))]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": output})


async def echo_body(scope, receive, send):
    status = 200
    headers = [(b"content-type", "text/plain")]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    more_body = True
    while more_body:
        message = await receive()
        body = message.get("body", b"")
        more_body = message.get("more_body", False)
        await send({"type": "http.response.body", "body": body, "more_body": more_body})


async def echo_headers(scope, receive, send):
    status = 200
    output = json.dumps(
        {"headers": [[k.decode(), v.decode()] for k, v in scope["headers"]]}
    ).encode("utf-8")
    headers = [(b"content-type", "text/plain"), (b"content-length", str(len(output)))]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": output})


async def raise_exc(scope, receive, send):
    raise RuntimeError()


async def raise_exc_after_response(scope, receive, send):
    status = 200
    output = b"Hello, World!"
    headers = [(b"content-type", "text/plain"), (b"content-length", str(len(output)))]

    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": output})
    raise RuntimeError()


@pytest.mark.anyio
async def test_asgi_transport():
    async with httpx.ASGITransport(app=hello_world) as transport:
        request = httpx.Request("GET", "http://www.example.com/")
        response = await transport.handle_async_request(request)
        await response.aread()
        assert response.status_code == 200
        assert response.content == b"Hello, World!"


@pytest.mark.anyio
async def test_asgi_transport_no_body():
    async with httpx.ASGITransport(app=echo_body) as transport:
        request = httpx.Request("GET", "http://www.example.com/")
        response = await transport.handle_async_request(request)
        await response.aread()
        assert response.status_code == 200
        assert response.content == b""


@pytest.mark.anyio
async def test_asgi():
    transport = httpx.ASGITransport(app=hello_world)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://www.example.org/")

    assert response.status_code == 200
    assert response.text == "Hello, World!"


@pytest.mark.anyio
async def test_asgi_urlencoded_path():
    transport = httpx.ASGITransport(app=echo_path)
    async with httpx.AsyncClient(transport=transport) as client:
        url = httpx.URL("http://www.example.org/").copy_with(path="/user@example.org")
        response = await client.get(url)

    assert response.status_code == 200
    assert response.json() == {"path": "/user@example.org"}


@pytest.mark.anyio
async def test_asgi_raw_path():
    transport = httpx.ASGITransport(app=echo_raw_path)
    async with httpx.AsyncClient(transport=transport) as client:
        url = httpx.URL("http://www.example.org/").copy_with(path="/user@example.org")
        response = await client.get(url)

    assert response.status_code == 200
    assert response.json() == {"raw_path": "/user@example.org"}


@pytest.mark.anyio
async def test_asgi_raw_path_should_not_include_querystring_portion():
    """
    See https://github.com/encode/httpx/issues/2810
    """
    transport = httpx.ASGITransport(app=echo_raw_path)
    async with httpx.AsyncClient(transport=transport) as client:
        url = httpx.URL("http://www.example.org/path?query")
        response = await client.get(url)

    assert response.status_code == 200
    assert response.json() == {"raw_path": "/path"}


@pytest.mark.anyio
async def test_asgi_upload():
    transport = httpx.ASGITransport(app=echo_body)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("http://www.example.org/", content=b"example")

    assert response.status_code == 200
    assert response.text == "example"


@pytest.mark.anyio
async def test_asgi_headers():
    transport = httpx.ASGITransport(app=echo_headers)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://www.example.org/")

    assert response.status_code == 200
    assert response.json() == {
        "headers": [
            ["host", "www.example.org"],
            ["accept", "*/*"],
            ["accept-encoding", "gzip, deflate, br, zstd"],
            ["connection", "keep-alive"],
            ["user-agent", f"python-httpx/{httpx.__version__}"],
        ]
    }


@pytest.mark.anyio
async def test_asgi_exc():
    transport = httpx.ASGITransport(app=raise_exc)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError):
            await client.get("http://www.example.org/")


@pytest.mark.anyio
async def test_asgi_exc_after_response():
    transport = httpx.ASGITransport(app=raise_exc_after_response)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError):
            await client.get("http://www.example.org/")


@pytest.mark.anyio
async def test_asgi_disconnect_after_response_complete():
    disconnect = False

    async def read_body(scope, receive, send):
        nonlocal disconnect

        status = 200
        headers = [(b"content-type", "text/plain")]

        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        more_body = True
        while more_body:
            message = await receive()
            more_body = message.get("more_body", False)

        await send({"type": "http.response.body", "body": b"", "more_body": False})

        # The ASGI spec says of the Disconnect message:
        # "Sent to the application when a HTTP connection is closed or if receive is
        # called after a response has been sent."
        # So if receive() is called again, the disconnect message should be received
        message = await receive()
        disconnect = message.get("type") == "http.disconnect"

    transport = httpx.ASGITransport(app=read_body)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("http://www.example.org/", content=b"example")

    assert response.status_code == 200
    assert disconnect


@pytest.mark.anyio
async def test_asgi_exc_no_raise():
    transport = httpx.ASGITransport(app=raise_exc, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://www.example.org/")

        assert response.status_code == 500
