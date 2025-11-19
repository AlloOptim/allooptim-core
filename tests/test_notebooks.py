"""Test Jupyter notebooks for execution errors and warnings."""

import warnings
from pathlib import Path

import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read
from nbformat.validator import ValidationError, validate


class TestJupyterNotebooks:
    """Test that all Jupyter notebooks execute without errors or warnings."""

    @pytest.fixture
    def notebook_paths(self):
        """Get all notebook files in the examples directory."""
        examples_dir = Path(__file__).parent.parent / "examples"
        return list(examples_dir.glob("*.ipynb"))

    def test_notebooks_are_valid(self, notebook_paths):
        """Test that all notebooks have valid structure."""
        for notebook_path in notebook_paths:
            with open(notebook_path, encoding="utf-8") as f:
                nb = read(f, as_version=4)

            # Validate notebook structure
            try:
                validate(nb)
            except ValidationError as e:
                pytest.fail(f"Notebook {notebook_path.name} has invalid structure: {str(e)}")

            # Check that notebook has cells
            assert len(nb.cells) > 0, f"Notebook {notebook_path.name} has no cells"

            # Check that notebook has at least one code cell
            code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
            assert len(code_cells) > 0, f"Notebook {notebook_path.name} has no code cells"

    def test_notebooks_execute_without_errors(self, notebook_paths):
        """Test that all notebooks execute without raising exceptions."""
        for notebook_path in notebook_paths:
            with self._notebook_execution_context(notebook_path):
                # Read the notebook
                with open(notebook_path, encoding="utf-8") as f:
                    nb = read(f, as_version=4)

                # Create execution preprocessor
                ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

                # Execute the notebook and catch any errors
                try:
                    ep.preprocess(nb)
                except Exception as e:
                    pytest.fail(f"Notebook {notebook_path.name} failed to execute: {str(e)}")

    def _notebook_execution_context(self, notebook_path):
        """Context manager to catch warnings during notebook execution."""
        return self.NotebookWarningCatcher(notebook_path)

    class NotebookWarningCatcher:
        """Context manager that catches warnings and converts them to test failures."""

        def __init__(self, notebook_path):
            """Initialize the warning catcher with notebook path."""
            self.notebook_path = notebook_path
            self.warnings_caught = []

        def __enter__(self):
            """Enter the context manager and set up warning capture."""
            # Set up warning capture
            self.original_showwarning = warnings.showwarning
            warnings.showwarning = self._capture_warning
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit the context manager and restore warning handler."""
            # Restore original warning handler
            warnings.showwarning = self.original_showwarning

            # Filter out system/infrastructure warnings that are not relevant
            relevant_warnings = self._filter_relevant_warnings()

            # If any relevant warnings were caught, fail the test
            if relevant_warnings:
                warning_messages = [str(w.message) for w in relevant_warnings]
                pytest.fail(f"Notebook {self.notebook_path.name} generated warnings: " f"{'; '.join(warning_messages)}")

        def _filter_relevant_warnings(self):
            """Filter out system warnings that are not relevant to notebook execution."""
            relevant_warnings = []

            for warning in self.warnings_caught:
                # Skip zmq/tornado event loop warnings (infrastructure)
                if "Proactor event loop does not implement add_reader" in str(warning.message):
                    continue
                # Skip asyncio selector warnings
                if "Registering an additional selector thread" in str(warning.message):
                    continue
                # Skip matplotlib backend warnings
                if "matplotlib" in str(warning.filename).lower() and "backend" in str(warning.message).lower():
                    continue
                # Skip seaborn warnings about deprecated parameters
                if "seaborn" in str(warning.filename).lower():
                    continue
                # Skip quantstats plotting warnings
                if "quantstats" in str(warning.filename).lower():
                    continue
                # Skip pandas future warnings
                if "pandas" in str(warning.filename).lower() and "FutureWarning" in str(warning.category.__name__):
                    continue

                # Include all other warnings
                relevant_warnings.append(warning)

            return relevant_warnings

        def _capture_warning(self, message, category, filename, lineno, file=None, line=None):
            """Capture warnings for later processing."""
            warning = warnings.WarningMessage(
                message=message, category=category, filename=filename, lineno=lineno, file=file, line=line
            )
            self.warnings_caught.append(warning)

            # Still show the warning to stderr
            self.original_showwarning(message, category, filename, lineno, file, line)
