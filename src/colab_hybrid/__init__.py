"""Colab Hybrid Runner: Zero-boilerplate Local CPU & Colab GPU orchestration."""

from .runner import hybrid_cell, export_to_notebook
from .handover import HandoverManager, HandoverPolicy, get_file_sha256

__version__ = "0.1.0"
__all__ = [
    "hybrid_cell",
    "export_to_notebook",
    "HandoverManager",
    "HandoverPolicy",
    "get_file_sha256",
]
