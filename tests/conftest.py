"""Shared test fixtures."""

from __future__ import annotations

import pytest

from yaharness.llm import MockLLMClient


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()
