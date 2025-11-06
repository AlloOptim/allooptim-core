Contributing
============

We welcome contributions to AlloOptim! This guide covers development setup,
coding standards, and contribution workflows.

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.12+
- Poetry (dependency management)
- Git

Installation
~~~~~~~~~~~~

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/AlloOptim/allooptim-core.git
      cd allooptim-core

2. **Install dependencies:**

   .. code-block:: bash

      poetry install --with dev,docs

3. **Set up pre-commit hooks:**

   .. code-block:: bash

      poetry run pre-commit install

4. **Run tests to verify setup:**

   .. code-block:: bash

      poetry run pytest

Coding Standards
----------------

Code Style
~~~~~~~~~~

We use `ruff` for code formatting and linting:

.. code-block:: bash

   # Format code
   poetry run ruff format allooptim/

   # Check for issues
   poetry run ruff check allooptim/

   # Fix auto-fixable issues
   poetry run ruff check --fix allooptim/

Docstrings
~~~~~~~~~~

Use Google-style docstrings for all public APIs:

.. code-block:: python

   def allocate(self, ds_mu, df_cov, **kwargs):
       """
       Compute optimal portfolio allocation.

       Args:
           ds_mu: Expected returns as pandas Series
           df_cov: Covariance matrix as pandas DataFrame
           **kwargs: Additional optimizer-specific parameters

       Returns:
           Portfolio weights as pandas Series

       Raises:
           ValueError: If inputs are invalid

       Examples:
           >>> optimizer = MeanVarianceOptimizer()
           >>> weights = optimizer.allocate(returns, cov_matrix)
       """
       pass

Type Hints
~~~~~~~~~~

Use comprehensive type hints:

.. code-block:: python

   from typing import Optional, Dict, Any
   import pandas as pd

   def optimize_portfolio(
       returns: pd.Series,
       covariance: pd.DataFrame,
       constraints: Optional[Dict[str, Any]] = None
   ) -> pd.Series:
       pass

Testing
-------

Unit Tests
~~~~~~~~~~

Add tests for new functionality:

.. code-block:: python

   # tests/test_my_optimizer.py
   import pytest
   import pandas as pd
   from allooptim.optimizer import MyOptimizer

   class TestMyOptimizer:
       def test_allocate_basic(self):
           # Test basic allocation
           mu = pd.Series([0.1, 0.05])
           cov = pd.DataFrame([[0.2, 0.1], [0.1, 0.15]])

           optimizer = MyOptimizer()
           weights = optimizer.allocate(mu, cov)

           assert weights.sum() == pytest.approx(1.0)
           assert all(weights >= 0)  # Long-only

       def test_allocate_edge_cases(self):
           # Test edge cases
           pass

Integration Tests
~~~~~~~~~~~~~~~~~

Test component interactions in ``tests/integration/``.

Documentation
-------------

Build Documentation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build docs locally
   cd docs
   poetry run make html

   # Check for warnings
   poetry run make html 2>&1 | grep -i warning

Update Documentation
~~~~~~~~~~~~~~~~~~~~

- Update docstrings when changing APIs
- Add examples for new features
- Update type hints
- Test documentation builds

Contribution Workflow
---------------------

1. **Fork the repository**

2. **Create a feature branch:**

   .. code-block:: bash

      git checkout -b feature/my-feature

3. **Make changes following coding standards**

4. **Add tests for new functionality**

5. **Update documentation**

6. **Run full test suite:**

   .. code-block:: bash

      poetry run pytest --cov=allooptim
      poetry run ruff check allooptim/
      cd docs && poetry run make html

7. **Commit with clear messages:**

   .. code-block:: bash

      git commit -m "feat: add new optimizer

      - Implements XYZ algorithm
      - Adds comprehensive tests
      - Updates documentation
      - Closes #123"

8. **Push and create pull request**

Pull Request Guidelines
-----------------------

**Title Format:**
- ``feat: add new optimizer`` (new features)
- ``fix: handle edge case in covariance`` (bug fixes)
- ``docs: update installation guide`` (documentation)
- ``refactor: simplify allocation logic`` (refactoring)

**Description:**
- Clear description of changes
- Link to related issues
- Screenshots for UI changes
- Test results

**Checklist:**
- [ ] Tests pass
- [ ] Code style checks pass
- [ ] Documentation builds
- [ ] Type hints complete
- [ ] Docstrings added/updated

Adding New Optimizers
---------------------

1. **Inherit from AbstractOptimizer:**

   .. code-block:: python

      from allooptim.optimizer.optimizer_interface import AbstractOptimizer

      class MyOptimizer(AbstractOptimizer):
          @property
          def name(self) -> str:
              return "MyOptimizer"

          def allocate(self, ds_mu, df_cov, **kwargs):
              # Implementation
              pass

2. **Add configuration class:**

   .. code-block:: python

      from pydantic import BaseModel

      class MyOptimizerConfig(BaseModel):
          parameter: float = 1.0

3. **Register in optimizer_list.py**

4. **Add comprehensive tests**

5. **Update documentation**

Adding Covariance Transformers
------------------------------

1. **Inherit from AbstractCovarianceTransformer:**

   .. code-block:: python

      from allooptim.covariance_transformer.transformer_interface import AbstractCovarianceTransformer

      class MyTransformer(AbstractCovarianceTransformer):
          def transform(self, df_cov, n_observations=None):
              # Implementation
              return transformed_cov

2. **Add to transformer_list.py**

3. **Add tests and documentation**

Reporting Issues
----------------

**Bug Reports:**
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details

**Feature Requests:**
- Clear use case description
- Proposed API
- Benefits and tradeoffs

**Questions:**
- Use GitHub Discussions
- Check documentation first
- Provide context

Code of Conduct
---------------

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional discourse

License
-------

By contributing, you agree to license your contributions under the MIT License.