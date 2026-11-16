from __future__ import annotations

import typing

import anyio

from .._models import Request, Response
from .._types import AsyncByteStream
from .base import AsyncBaseTransport

if typing.TYPE_CHECKING:  # pragma: no cover
    import asyncio

    import trio

    Event = typing.Union[asyncio.Event, trio.Event]


_Message = typing.MutableMapping[str, typing.Any]
_Receive = typing.Callable[[], typing.Awaitable[_Message]]
_Send = typing.Callable[
    [typing.MutableMapping[str, typing.Any]], typing.Awaitable[None]
]
_ASGIApp = typing.Callable[
    [typing.MutableMapping[str, typing.Any], _Receive, _Send], typing.Awaitable[None]
]

__all__ = ["ASGITransport"]


def is_running_trio() -> bool:
    try:
        # sniffio is a dependency of trio.

        # See https://github.com/python-trio/trio/issues/2802
        import sniffio

        if sniffio.current_async_library() == "trio":
            return True
    except ImportError:  # pragma: nocover
        pass

    return False


def create_event() -> Event:
    if is_running_trio():
        import trio

        return trio.Event()

    import asyncio

    return asyncio.Event()


class ASGIResponseStream(AsyncByteStream):
    def __init__(
        self,
        receive_stream: anyio.abc.ObjectReceiveStream[bytes],
        response_complete: Event,
        app_done: Event,
        cancel_app: typing.Callable[[], None],
        app_error: typing.Callable[[], Exception | None],
        on_close: typing.Callable[[ASGIResponseStream], None],
    ) -> None:
        self._receive_stream = receive_stream
        self._response_complete = response_complete
        self._app_done = app_done
        self._cancel_app = cancel_app
        self._app_error = app_error
        self._on_close = on_close
        self._exhausted = False
        self._closed = False

    def __aiter__(self) -> ASGIResponseStream:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._receive_stream.receive()
        except anyio.EndOfStream:
            self._exhausted = True
            await self.aclose()
            if error := self._app_error():
                raise error
            raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._response_complete.set()
        if not self._exhausted:
            self._cancel_app()
        # Cleanup is necessary even when the enclosing response is cancelled.
        try:
            with anyio.CancelScope(shield=True):
                await self._receive_stream.aclose()
                await self._app_done.wait()
        finally:
            self._on_close(self)


class ASGITransport(AsyncBaseTransport):
    """
    A custom AsyncTransport that handles sending requests directly to an ASGI app.

    ```python
    transport = httpx.ASGITransport(
        app=app,
        root_path="/submount",
        client=("1.2.3.4", 123)
    )
    client = httpx.AsyncClient(transport=transport)
    ```

    Arguments:

    * `app` - The ASGI application.
    * `raise_app_exceptions` - Boolean indicating if exceptions in the application
       should be raised. Default to `True`. Can be set to `False` for use cases
       such as testing the content of a client 500 response.
    * `root_path` - The root path on which the ASGI application should be mounted.
    * `client` - A two-tuple indicating the client IP and port of incoming requests.
    ```
    """

    def __init__(
        self,
        app: _ASGIApp,
        raise_app_exceptions: bool = True,
        root_path: str = "",
        client: tuple[str, int] = ("127.0.0.1", 123),
    ) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions
        self.root_path = root_path
        self.client = client
        self._active_streams: set[ASGIResponseStream] = set()

    async def handle_async_request(
        self,
        request: Request,
    ) -> Response:
        request_stream = request.stream
        assert isinstance(request_stream, AsyncByteStream)

        # ASGI scope.
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": self.client,
            "root_path": self.root_path,
        }

        # Request.
        request_body_chunks = request_stream.__aiter__()
        request_complete = False

        # Response.
        status_code = None
        response_headers = None
        response_started = False
        response_complete = create_event()
        response_started_event = create_event()
        app_done = create_event()
        app_error: Exception | None = None
        cancel_app: list[typing.Callable[[], None]] = [lambda: None]
        cancel_requested = False
        send_stream: anyio.abc.ObjectSendStream[bytes]
        receive_stream: anyio.abc.ObjectReceiveStream[bytes]
        send_stream, receive_stream = anyio.create_memory_object_stream(1)

        def cancel_runner() -> None:
            nonlocal cancel_requested
            cancel_requested = True
            cancel_app[0]()

        # ASGI callables.

        async def receive() -> dict[str, typing.Any]:
            nonlocal request_complete

            if request_complete:
                await response_complete.wait()
                return {"type": "http.disconnect"}

            try:
                body = await request_body_chunks.__anext__()
            except StopAsyncIteration:
                request_complete = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": body, "more_body": True}

        async def send(message: typing.MutableMapping[str, typing.Any]) -> None:
            nonlocal status_code, response_headers, response_started

            if message["type"] == "http.response.start":
                assert not response_started

                status_code = message["status"]
                response_headers = message.get("headers", [])
                response_started = True
                response_started_event.set()

            elif message["type"] == "http.response.body":
                assert not response_complete.is_set()
                body = message.get("body", b"")
                more_body = message.get("more_body", False)

                if body and request.method != "HEAD":
                    await send_stream.send(body)

                if not more_body:
                    response_complete.set()

        async def run_app() -> None:
            nonlocal app_error, status_code, response_headers
            try:
                if is_running_trio():
                    import trio

                    with trio.CancelScope() as app_cancel_scope:
                        cancel_app[0] = app_cancel_scope.cancel
                        if cancel_requested:
                            app_cancel_scope.cancel()
                        await self.app(scope, receive, send)
                else:
                    await self.app(scope, receive, send)
                if not response_complete.is_set():
                    raise AssertionError(
                        "ASGI application did not finish response body"
                    )
                if status_code is None:
                    raise AssertionError("ASGI application did not start a response")
            except Exception as exc:
                if self.raise_app_exceptions:
                    app_error = exc
                if status_code is None:
                    status_code = 500
                    response_headers = []
            finally:
                with anyio.CancelScope(shield=True):
                    response_complete.set()
                    response_started_event.set()
                    request_close = getattr(request_body_chunks, "aclose", None)
                    try:
                        for close in (
                            send_stream.aclose,
                            request_close,
                            request_stream.aclose,
                        ):
                            if close is not None:
                                try:
                                    await close()
                                except Exception as exc:
                                    if self.raise_app_exceptions and app_error is None:
                                        app_error = exc
                    finally:
                        app_done.set()

        # Use a separate task so response headers and body chunks can be sent
        # before an ASGI application returns (for example, SSE endpoints).
        if is_running_trio():
            import trio

            trio.lowlevel.spawn_system_task(run_app)
        else:
            import asyncio

            task = asyncio.create_task(run_app())
            # A task cancelled before its first step never runs its finally block.
            task.add_done_callback(lambda _: app_done.set())

            def cancel_task() -> None:
                task.cancel()

            cancel_app[0] = cancel_task

        try:
            await response_started_event.wait()
        except BaseException:
            cancel_runner()
            with anyio.CancelScope(shield=True):
                response_complete.set()
                await send_stream.aclose()
                await receive_stream.aclose()
                await app_done.wait()
            raise

        if app_error is not None:
            await receive_stream.aclose()
            await app_done.wait()
            raise app_error
        assert status_code is not None
        assert response_headers is not None

        stream = ASGIResponseStream(
            receive_stream,
            response_complete,
            app_done,
            cancel_runner,
            lambda: app_error,
            self._active_streams.discard,
        )
        self._active_streams.add(stream)
        return Response(status_code, headers=response_headers, stream=stream)

    async def aclose(self) -> None:
        for stream in tuple(self._active_streams):
            await stream.aclose()
        self._active_streams.clear()
