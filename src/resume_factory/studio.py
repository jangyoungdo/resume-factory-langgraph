"""Offline graph entrypoint for LangGraph Studio."""

from .agents import DeterministicBackend
from .graph import build_graph

graph = build_graph(DeterministicBackend())

