"""Shared pytest fixtures and configuration for all tests."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_path = temp_dir / "test_file.txt"
    temp_path.write_text("test content")
    yield temp_path


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "debug": True,
        "max_iterations": 100,
        "timeout": 30,
        "output_dir": "/tmp/test_output",
        "model_name": "test_model",
    }


@pytest.fixture
def sample_puzzle_data() -> Dict[str, Any]:
    """Provide sample puzzle data for testing."""
    return {
        "name": "test_puzzle",
        "description": "A test puzzle for unit testing",
        "input": [1, 2, 3, 4, 5],
        "expected_output": 15,
        "difficulty": "easy",
        "tags": ["math", "sum"],
    }


@pytest.fixture
def sample_code_snippet() -> str:
    """Provide a sample code snippet for testing."""
    return """
def solve(input_data):
    '''Solve the puzzle by summing all numbers.'''
    return sum(input_data)
"""


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "TEST_MODE": "true",
        "LOG_LEVEL": "DEBUG",
        "OUTPUT_FORMAT": "json",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def generators_path(project_root: Path) -> Path:
    """Return the path to the generators module."""
    return project_root / "generators"


@pytest.fixture
def solvers_path(project_root: Path) -> Path:
    """Return the path to the solvers module."""
    return project_root / "solvers"


@pytest.fixture
def clean_imports():
    """Clean up imports to ensure test isolation."""
    modules_to_remove = []
    for module in sys.modules:
        if module.startswith(("generators", "solvers")):
            modules_to_remove.append(module)
    
    yield
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed for reproducible tests."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    yield
    
    # Reset after test
    random.seed()
    np.random.seed()


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that should be run less frequently"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with "slow" in their name
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)