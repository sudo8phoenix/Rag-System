"""UI entrypoints for web interfaces."""

from typing import Any


def create_gradio_app(*args: Any, **kwargs: Any):
	"""Lazily import Gradio app factory to avoid run-module warnings."""

	from .gradio_app import create_gradio_app as _create_gradio_app

	return _create_gradio_app(*args, **kwargs)

__all__ = ["create_gradio_app"]
