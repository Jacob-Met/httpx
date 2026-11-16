from __future__ import annotations

from collections.abc import Iterator
from http.cookiejar import Cookie, CookieJar

import pytest

import httpx


def get_and_set_cookies(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/echo_cookies":
        data = {"cookies": request.headers.get("cookie")}
        return httpx.Response(200, json=data)
    elif request.url.path == "/set_cookie":
        return httpx.Response(200, headers={"set-cookie": "example-name=example-value"})
    else:
        raise NotImplementedError()  # pragma: no cover


@pytest.mark.parametrize("client_cls", [httpx.Client, httpx.AsyncClient])
def test_build_request_cookie_snapshot(
    client_cls: type[httpx.Client] | type[httpx.AsyncClient],
) -> None:
    client = client_cls(cookies={"name": "before"})
    first = client.build_request("GET", "https://example.org/echo_cookies")
    client.cookies.set("name", "after")
    second = client.build_request("GET", "https://example.org/echo_cookies")

    assert first.headers["cookie"] == "name=before"
    assert second.headers["cookie"] == "name=after"
    assert client.cookies["name"] == "after"


def test_set_cookie() -> None:
    """
    Send a request including a cookie.
    """
    url = "http://example.org/echo_cookies"
    cookies = {"example-name": "example-value"}

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_per_request_cookie_is_deprecated() -> None:
    """
    Sending a request including a per-request cookie is deprecated.
    """
    url = "http://example.org/echo_cookies"
    cookies = {"example-name": "example-value"}

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    with pytest.warns(DeprecationWarning):
        response = client.get(url, cookies=cookies)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_cookie_with_cookiejar() -> None:
    """
    Send a request including a cookie, using a `CookieJar` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = CookieJar()
    cookie = Cookie(
        version=0,
        name="example-name",
        value="example-value",
        port=None,
        port_specified=False,
        domain="",
        domain_specified=False,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={"HttpOnly": ""},
        rfc2109=False,
    )
    cookies.set_cookie(cookie)

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_setting_client_cookies_to_cookiejar() -> None:
    """
    Send a request including a cookie, using a `CookieJar` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = CookieJar()
    cookie = Cookie(
        version=0,
        name="example-name",
        value="example-value",
        port=None,
        port_specified=False,
        domain="",
        domain_specified=False,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={"HttpOnly": ""},
        rfc2109=False,
    )
    cookies.set_cookie(cookie)

    client = httpx.Client(
        cookies=cookies, transport=httpx.MockTransport(get_and_set_cookies)
    )
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_set_cookie_with_cookies_model() -> None:
    """
    Send a request including a cookie, using a `Cookies` instance.
    """

    url = "http://example.org/echo_cookies"
    cookies = httpx.Cookies()
    cookies["example-name"] = "example-value"

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    client.cookies = cookies
    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_get_cookie() -> None:
    url = "http://example.org/set_cookie"

    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))
    response = client.get(url)

    assert response.status_code == 200
    assert response.cookies["example-name"] == "example-value"
    assert client.cookies["example-name"] == "example-value"


def test_cookie_persistence() -> None:
    """
    Ensure that Client instances persist cookies between requests.
    """
    client = httpx.Client(transport=httpx.MockTransport(get_and_set_cookies))

    response = client.get("http://example.org/echo_cookies")
    assert response.status_code == 200
    assert response.json() == {"cookies": None}

    response = client.get("http://example.org/set_cookie")
    assert response.status_code == 200
    assert response.cookies["example-name"] == "example-value"
    assert client.cookies["example-name"] == "example-value"

    response = client.get("http://example.org/echo_cookies")
    assert response.status_code == 200
    assert response.json() == {"cookies": "example-name=example-value"}


def test_client_build_request_reuses_cookie_jar_without_copying() -> None:
    # The underlying CookieJar expires cookies once during header preparation,
    # but the client must not iterate and copy its entire jar beforehand.
    class CountingJar(CookieJar):
        def __init__(self) -> None:
            super().__init__()
            self.traversals = 0

        def __iter__(self) -> Iterator[Cookie]:
            self.traversals += 1
            return super().__iter__()

    client = httpx.Client(cookies={"session": "before"})
    counting = CountingJar()
    for cookie in client.cookies.jar:
        counting.set_cookie(cookie)
    client.cookies.jar = counting

    first = client.build_request("GET", "https://example.org/path")
    assert first.headers["cookie"] == "session=before"
    assert counting.traversals <= 1

    client.cookies.set("session", "after")
    traversals_before = counting.traversals
    second = client.build_request("GET", "https://example.org/path")
    assert second.headers["cookie"] == "session=after"
    assert first.headers["cookie"] == "session=before"
    assert counting.traversals - traversals_before <= 1

    # An empty client jar must retain the zero-cookie fast path.
    client.cookies.clear()
    traversals_before = counting.traversals
    assert "cookie" not in client.build_request("GET", "https://example.org/").headers
    assert counting.traversals == traversals_before


def test_per_request_cookies_override_without_persisting() -> None:
    client = httpx.Client(cookies={"session": "client", "other": "retained"})
    request = client.build_request(
        "GET",
        "https://example.org/",
        cookies={"session": "request", "new": "only-once"},
    )
    assert set(request.headers["cookie"].split("; ")) == {
        "session=request",
        "other=retained",
        "new=only-once",
    }
    assert client.cookies["session"] == "client"
    assert "new" not in client.cookies
