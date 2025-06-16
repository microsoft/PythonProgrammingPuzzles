"""Validation tests to ensure the testing infrastructure is properly configured."""
import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Verify that pytest is installed and working."""
        assert True, "Basic pytest assertion should work"
    
    @pytest.mark.unit
    def test_project_imports(self):
        """Verify that project modules can be imported."""
        # Test importing from generators
        try:
            import generators
            assert hasattr(generators, '__file__')
        except ImportError:
            pytest.fail("Failed to import generators module")
        
        # Test importing from solvers
        try:
            import solvers
            assert hasattr(solvers, '__file__')
        except ImportError:
            pytest.fail("Failed to import solvers module")
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_config, project_root):
        """Verify that custom fixtures are available."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert isinstance(mock_config, dict)
        assert "debug" in mock_config
        
        assert isinstance(project_root, Path)
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()
    
    @pytest.mark.unit
    def test_markers_configured(self, request):
        """Verify that custom markers are properly configured."""
        markers = request.config.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names
    
    @pytest.mark.unit
    def test_coverage_configured(self):
        """Verify that coverage is properly configured."""
        try:
            import coverage
            assert hasattr(coverage, '__version__')
        except ImportError:
            pytest.fail("Coverage module not available")
    
    @pytest.mark.unit
    def test_temp_file_fixture(self, temp_file):
        """Test that temp_file fixture works correctly."""
        assert temp_file.exists()
        assert temp_file.read_text() == "test content"
    
    @pytest.mark.unit
    def test_sample_fixtures(self, sample_puzzle_data, sample_code_snippet):
        """Test that sample data fixtures work correctly."""
        assert isinstance(sample_puzzle_data, dict)
        assert sample_puzzle_data["name"] == "test_puzzle"
        assert sample_puzzle_data["expected_output"] == 15
        
        assert isinstance(sample_code_snippet, str)
        assert "def solve" in sample_code_snippet
        assert "return sum(input_data)" in sample_code_snippet
    
    @pytest.mark.unit
    def test_environment_mock(self, mock_environment_variables):
        """Test that environment variable mocking works."""
        import os
        assert os.environ.get("TEST_MODE") == "true"
        assert os.environ.get("LOG_LEVEL") == "DEBUG"
        assert os.environ.get("OUTPUT_FORMAT") == "json"
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True, "Integration test marker should work"
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True, "Slow test marker should work"


class TestPytestFeatures:
    """Test various pytest features to ensure they work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_value,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
    ])
    def test_parametrize(self, input_value, expected):
        """Test that parametrize decorator works."""
        assert input_value ** 2 == expected
    
    @pytest.mark.unit
    def test_pytest_raises(self):
        """Test that pytest.raises works correctly."""
        with pytest.raises(ValueError, match="test error"):
            raise ValueError("test error")
    
    @pytest.mark.unit
    def test_pytest_warns(self):
        """Test that pytest.warns works correctly."""
        import warnings
        
        with pytest.warns(UserWarning, match="test warning"):
            warnings.warn("test warning", UserWarning)
    
    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform == "win32", reason="Skip on Windows")
    def test_skipif(self):
        """Test that skipif decorator works."""
        assert sys.platform != "win32"
    
    @pytest.mark.unit
    def test_monkeypatch(self, monkeypatch):
        """Test that monkeypatch fixture works."""
        def mock_function(path):
            return "mocked"
        
        import os
        monkeypatch.setattr(os.path, "exists", mock_function)
        assert os.path.exists("/fake/path") == "mocked"