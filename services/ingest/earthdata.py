"""
Helpers for authenticated access to NASA Earthdata-protected endpoints.

This module wraps `requests.Session` so that the Authorization header is
preserved across redirects between the data host and `urs.earthdata.nasa.gov`,
matching NASA’s recommended pattern.
"""

from __future__ import annotations

import logging
import os
import time
from netrc import NetrcParseError, netrc
from pathlib import Path
from typing import Optional, Tuple

import requests

AUTH_HOST = "urs.earthdata.nasa.gov"
LOGGER = logging.getLogger(__name__)


def _load_netrc_credentials(host: str) -> Optional[Tuple[str, str]]:
    """
    Load (login, password) from either ~/.netrc or a project-local .netrc.
    """
    for candidate in (Path.home() / ".netrc", Path(".netrc")):
        if not candidate.exists():
            continue
        try:
            auth = netrc(candidate).authenticators(host)
        except (OSError, NetrcParseError) as exc:
            LOGGER.warning("Failed to parse %s for Earthdata credentials: %s", candidate, exc)
            continue
        if auth is None:
            continue
        login, _, password = auth
        if login and password:
            return login, password
    return None


class EarthdataSession(requests.Session):
    """
    Requests session that attaches Earthdata Login credentials.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None) -> None:
        super().__init__()
        user = username or os.getenv("NASA_EARTHDATA_USERNAME")
        pwd = password or os.getenv("NASA_EARTHDATA_PASSWORD")
        if not (user and pwd):
            creds = _load_netrc_credentials(AUTH_HOST)
            if creds:
                user = user or creds[0]
                pwd = pwd or creds[1]

        if not (user and pwd):
            raise RuntimeError(
                "NASA Earthdata credentials not configured. Set NASA_EARTHDATA_USERNAME/PASSWORD or populate .netrc."
            )

        self.auth = (user, pwd)

    def rebuild_auth(self, prepared_request: requests.PreparedRequest, response: requests.Response) -> None:
        """
        Keep Authorization when redirecting within Earthdata, drop otherwise.
        """
        if "Authorization" not in prepared_request.headers:
            return

        original = requests.utils.urlparse(response.request.url)
        redirect = requests.utils.urlparse(prepared_request.url)
        if (
            original.hostname != redirect.hostname
            and redirect.hostname != AUTH_HOST
            and original.hostname != AUTH_HOST
        ):
            del prepared_request.headers["Authorization"]


def download_file(url: str, dest: Path, session: Optional[EarthdataSession] = None, retries: int = 5) -> Path:
    """Download a file requiring Earthdata auth to `dest`, with simple retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or EarthdataSession()
    backoff = 5.0
    last_err: Optional[Exception] = None
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(dest)
            return dest
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_err = exc
            LOGGER.warning("Download failed (%s/%s) for %s: %s", attempt, retries, url, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as exc:  # pragma: no cover - bubble up unexpected errors
            last_err = exc
            break
    if tmp_path.exists():  # clean partial
        tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}")
