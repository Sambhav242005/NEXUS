"""Shared HTTP transport for voice providers (stdlib only).

One injectable transport for ASR + TTS so unit tests never touch the
network. Errors are typed and never carry key material.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Protocol


class VoiceProviderError(RuntimeError):
    """Typed voice transport/API failure. Never carries key material."""


@dataclass
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    def post_multipart(
        self,
        url: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse: ...

    def post_json(
        self,
        url: str,
        *,
        payload: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse: ...


def encode_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    boundary: str,
) -> bytes:
    buf = bytearray()
    for key, value in fields.items():
        buf += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
            f"\r\n\r\n{value}\r\n"
        ).encode("utf-8")
    for key, (filename, data, content_type) in files.items():
        buf += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode("utf-8")
        buf += data + b"\r\n"
    buf += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(buf)


def resolve_key(cfg_api_key: str | None, env_name: str, provider: str) -> str:
    if cfg_api_key:
        return cfg_api_key
    value = os.environ.get(env_name, "")
    if not value:
        raise VoiceProviderError(
            f"{provider} needs an API key: set {env_name} or pass api_key explicitly"
        )
    return value


def check_status(url: str, response: HttpResponse) -> None:
    if 200 <= response.status < 300:
        return
    detail = response.body.decode("utf-8", "replace")[:200]
    raise VoiceProviderError(f"POST {url} -> HTTP {response.status}: {detail}")


class UrllibTransport:
    """Default stdlib transport. Raises VoiceProviderError, never leaks headers."""

    def post_multipart(
        self,
        url: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        boundary = uuid.uuid4().hex
        return self._post(
            url,
            body=encode_multipart(fields, files, boundary),
            headers={
                **headers,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            timeout=timeout,
        )

    def post_json(
        self,
        url: str,
        *,
        payload: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        return self._post(
            url,
            body=payload,
            headers={**headers, "Content-Type": "application/json"},
            timeout=timeout,
        )

    @staticmethod
    def _post(
        url: str, *, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(response.status, response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise VoiceProviderError(
                f"POST {url} -> HTTP {exc.code}: {detail}"
            ) from exc
        except OSError as exc:
            raise VoiceProviderError(f"POST {url} failed: {exc}") from exc
