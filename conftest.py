"""
pytest configuration for the GhostSignal project.
Registers custom marks so they do not produce PytestUnknownMarkWarning.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit: fast unit tests that do not require trained model artifacts",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests that load real model artifacts from models/ — "
        "skipped automatically when artifacts are absent",
    )
