import ssl
import typing
from pathlib import Path

import certifi
import pytest

import httpx
from httpx._config import _DeferredDefaultSSLContext


def test_load_ssl_config():
    context = httpx.create_ssl_context()
    assert context.verify_mode == ssl.VerifyMode.CERT_REQUIRED
    assert context.check_hostname is True


def test_load_ssl_config_verify_non_existing_file():
    with pytest.raises(IOError):
        context = httpx.create_ssl_context()
        context.load_verify_locations(cafile="/path/to/nowhere")


def test_load_ssl_with_keylog(monkeypatch: typing.Any) -> None:
    monkeypatch.setenv("SSLKEYLOGFILE", "test")
    context = httpx.create_ssl_context()
    assert context.keylog_filename == "test"


def test_load_ssl_config_verify_existing_file():
    context = httpx.create_ssl_context()
    context.load_verify_locations(capath=certifi.where())
    assert context.verify_mode == ssl.VerifyMode.CERT_REQUIRED
    assert context.check_hostname is True


def test_load_ssl_config_verify_directory():
    context = httpx.create_ssl_context()
    context.load_verify_locations(capath=Path(certifi.where()).parent)
    assert context.verify_mode == ssl.VerifyMode.CERT_REQUIRED
    assert context.check_hostname is True


def test_load_ssl_config_cert_and_key(cert_pem_file, cert_private_key_file):
    context = httpx.create_ssl_context()
    context.load_cert_chain(cert_pem_file, cert_private_key_file)
    assert context.verify_mode == ssl.VerifyMode.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.mark.parametrize("password", [b"password", "password"])
def test_load_ssl_config_cert_and_encrypted_key(
    cert_pem_file, cert_encrypted_private_key_file, password
):
    context = httpx.create_ssl_context()
    context.load_cert_chain(cert_pem_file, cert_encrypted_private_key_file, password)
    assert context.verify_mode == ssl.VerifyMode.CERT_REQUIRED
    assert context.check_hostname is True


def test_load_ssl_config_cert_and_key_invalid_password(
    cert_pem_file, cert_encrypted_private_key_file
):
    with pytest.raises(ssl.SSLError):
        context = httpx.create_ssl_context()
        context.load_cert_chain(
            cert_pem_file, cert_encrypted_private_key_file, "password1"
        )


def test_load_ssl_config_cert_without_key_raises(cert_pem_file):
    with pytest.raises(ssl.SSLError):
        context = httpx.create_ssl_context()
        context.load_cert_chain(cert_pem_file)


def test_load_ssl_config_no_verify():
    context = httpx.create_ssl_context(verify=False)
    assert context.verify_mode == ssl.VerifyMode.CERT_NONE
    assert context.check_hostname is False


def test_SSLContext_with_get_request(server, cert_pem_file):
    context = httpx.create_ssl_context()
    context.load_verify_locations(cert_pem_file)
    response = httpx.get(server.url, verify=context)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "transport_cls", [httpx.HTTPTransport, httpx.AsyncHTTPTransport]
)
def test_transport_defers_default_ssl_trust_roots(transport_cls):
    first = transport_cls(trust_env=False)
    second = transport_cls(trust_env=False)
    first_context = first._pool._ssl_context
    second_context = second._pool._ssl_context

    assert isinstance(first_context, _DeferredDefaultSSLContext)
    assert isinstance(second_context, _DeferredDefaultSSLContext)
    assert first_context is not second_context
    assert first_context._lazy_context is None
    assert second_context._lazy_context is None

    # The protocol choice may be set before the CA bundle is loaded.
    first_context.set_alpn_protocols(["http/1.1"])
    first_context.wrap_bio(
        ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="example.org"
    )
    assert first_context._lazy_context is not None
    assert first_context._lazy_context.cert_store_stats()["x509"] > 0
    assert second_context._lazy_context is None


@pytest.mark.parametrize(
    "transport_cls", [httpx.HTTPTransport, httpx.AsyncHTTPTransport]
)
def test_transport_respects_explicit_ssl_context(transport_cls):
    custom = ssl.create_default_context(cafile=certifi.where())
    transport = transport_cls(verify=custom)
    assert transport._pool._ssl_context is custom


@pytest.mark.parametrize(
    "transport_cls", [httpx.HTTPTransport, httpx.AsyncHTTPTransport]
)
def test_transport_respects_ssl_env_configuration(
    transport_cls, monkeypatch, cert_pem_file, tmp_path
):
    monkeypatch.setenv("SSL_CERT_FILE", cert_pem_file)
    transport = transport_cls()
    context = transport._pool._ssl_context
    assert not hasattr(context, "_lazy_context")
    assert context.cert_store_stats()["x509"] > 0

    # trust_env=False must not inherit the environment's custom trust roots.
    untrusted = transport_cls(trust_env=False)._pool._ssl_context
    assert isinstance(untrusted, _DeferredDefaultSSLContext)
    assert untrusted._lazy_context is None

    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "keys.log"))
    keylog_context = transport_cls(trust_env=False)._pool._ssl_context
    assert not hasattr(keylog_context, "_lazy_context")
    assert keylog_context.keylog_filename == str(tmp_path / "keys.log")


def test_http_only_client_does_not_load_default_ssl_roots(server):
    with httpx.Client(trust_env=False) as client:
        assert isinstance(client._transport, httpx.HTTPTransport)
        context = client._transport._pool._ssl_context
        assert isinstance(context, _DeferredDefaultSSLContext)
        assert context._lazy_context is None
        assert client.get(server.url).status_code == 200
        assert context._lazy_context is None


@pytest.mark.anyio
async def test_http_only_async_client_does_not_load_default_ssl_roots(server):
    async with httpx.AsyncClient(trust_env=False) as client:
        assert isinstance(client._transport, httpx.AsyncHTTPTransport)
        context = client._transport._pool._ssl_context
        assert isinstance(context, _DeferredDefaultSSLContext)
        assert context._lazy_context is None
        assert (await client.get(server.url)).status_code == 200
        assert context._lazy_context is None


def test_limits_repr():
    limits = httpx.Limits(max_connections=100)
    expected = (
        "Limits(max_connections=100, max_keepalive_connections=None,"
        " keepalive_expiry=5.0)"
    )
    assert repr(limits) == expected


def test_limits_eq():
    limits = httpx.Limits(max_connections=100)
    assert limits == httpx.Limits(max_connections=100)


def test_timeout_eq():
    timeout = httpx.Timeout(timeout=5.0)
    assert timeout == httpx.Timeout(timeout=5.0)


def test_timeout_all_parameters_set():
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    assert timeout == httpx.Timeout(timeout=5.0)


def test_timeout_from_nothing():
    timeout = httpx.Timeout(None)
    assert timeout.connect is None
    assert timeout.read is None
    assert timeout.write is None
    assert timeout.pool is None


def test_timeout_from_none():
    timeout = httpx.Timeout(timeout=None)
    assert timeout == httpx.Timeout(None)


def test_timeout_from_one_none_value():
    timeout = httpx.Timeout(None, read=None)
    assert timeout == httpx.Timeout(None)


def test_timeout_from_one_value():
    timeout = httpx.Timeout(None, read=5.0)
    assert timeout == httpx.Timeout(timeout=(None, 5.0, None, None))


def test_timeout_from_one_value_and_default():
    timeout = httpx.Timeout(5.0, pool=60.0)
    assert timeout == httpx.Timeout(timeout=(5.0, 5.0, 5.0, 60.0))


def test_timeout_missing_default():
    with pytest.raises(ValueError):
        httpx.Timeout(pool=60.0)


def test_timeout_from_tuple():
    timeout = httpx.Timeout(timeout=(5.0, 5.0, 5.0, 5.0))
    assert timeout == httpx.Timeout(timeout=5.0)


def test_timeout_from_config_instance():
    timeout = httpx.Timeout(timeout=5.0)
    assert httpx.Timeout(timeout) == httpx.Timeout(timeout=5.0)


def test_timeout_repr():
    timeout = httpx.Timeout(timeout=5.0)
    assert repr(timeout) == "Timeout(timeout=5.0)"

    timeout = httpx.Timeout(None, read=5.0)
    assert repr(timeout) == "Timeout(connect=None, read=5.0, write=None, pool=None)"


def test_proxy_from_url():
    proxy = httpx.Proxy("https://example.com")

    assert str(proxy.url) == "https://example.com"
    assert proxy.auth is None
    assert proxy.headers == {}
    assert repr(proxy) == "Proxy('https://example.com')"


def test_proxy_with_auth_from_url():
    proxy = httpx.Proxy("https://username:password@example.com")

    assert str(proxy.url) == "https://example.com"
    assert proxy.auth == ("username", "password")
    assert proxy.headers == {}
    assert repr(proxy) == "Proxy('https://example.com', auth=('username', '********'))"


def test_invalid_proxy_scheme():
    with pytest.raises(ValueError):
        httpx.Proxy("invalid://example.com")
