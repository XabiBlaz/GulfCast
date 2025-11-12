# services/ingest/iri.py
from __future__ import annotations
import os
import requests
import xarray as xr

def open_iridl_dataset(url: str) -> xr.Dataset:
    username = os.getenv("IRIDL_USERNAME")
    password = os.getenv("IRIDL_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set IRIDL_USERNAME and IRIDL_PASSWORD for iridl access.")

    session = requests.Session()
    session.auth = (username, password)

    return xr.open_dataset(url, engine="pydap", backend_kwargs={"session": session})
