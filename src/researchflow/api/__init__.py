"""HTTP API package. Use ``researchflow.api.app:app`` for uvicorn."""

from importlib import import_module

_app_module = import_module("researchflow.api.app")
app = _app_module.app

__all__ = ["app"]
