"""Root-level backward compatibility wrapper."""
from src.colab_hybrid.runner import hybrid_cell, export_to_notebook, REGISTERED_CELLS

__all__ = ["hybrid_cell", "export_to_notebook", "REGISTERED_CELLS"]
