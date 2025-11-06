# Release Checklist for v0.3.0

## Pre-Release Setup (One-Time Only)

### ✅ Step 1: Configure PyPI Trusted Publishers

#### TestPyPI
1. Visit: https://test.pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - PyPI Project Name: `allooptim`
   - Owner: `AlloOptim`
   - Repository name: `allooptim-core`
   - Workflow name: `publish.yml`
   - Environment name: `testpypi`
4. Click "Add"

#### PyPI (Production)
1. Visit: https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - PyPI Project Name: `allooptim`
   - Owner: `AlloOptim`
   - Repository name: `allooptim-core`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
4. Click "Add"

### ✅ Step 2: Configure GitHub Environments

1. Visit: https://github.com/AlloOptim/allooptim-core/settings/environments
2. Create `testpypi` environment:
   - Click "New environment"
   - Name: `testpypi`
   - No protection rules needed
   - Click "Configure environment"
3. Create `pypi` environment:
   - Click "New environment"
   - Name: `pypi`
   - ✅ Check "Required reviewers"
   - Add yourself as a reviewer
   - (Optional) Set wait timer to 5 minutes
   - Click "Configure environment"

## Release Steps (v0.3.0)

### Local Preparation

- [x] CHANGELOG.md created
- [x] pyproject.toml updated with metadata
- [x] Version is 0.3.0 in pyproject.toml
- [ ] Run tests: `poetry run pytest tests/`
- [ ] Run linting: `poetry run ruff check allooptim`

### Git Operations

```powershell
# 1. Stage all changes
git add .

# 2. Commit changes
git commit -m "chore: prepare release v0.3.0

- Add CHANGELOG.md
- Update pyproject.toml with PyPI metadata
- Add publishing documentation"

# 3. Push to main
git push origin main

# 4. Create and push tag
git tag -a v0.3.0 -m "Release v0.3.0 - Initial public release"
git push origin v0.3.0
```

### Monitor Release

1. **Watch GitHub Actions**: https://github.com/AlloOptim/allooptim-core/actions
   - Build job should complete
   - TestPyPI publish should succeed automatically
   - PyPI publish will wait for manual approval

2. **Check TestPyPI**: https://test.pypi.org/project/allooptim/
   - Verify package appears
   - Check metadata looks correct

3. **Approve PyPI Release**:
   - Go to: https://github.com/AlloOptim/allooptim-core/actions
   - Click on the running workflow
   - Review and approve the PyPI deployment

4. **Verify PyPI**: https://pypi.org/project/allooptim/
   - Package should appear after approval
   - Check installation works: `pip install allooptim`

5. **GitHub Release**: https://github.com/AlloOptim/allooptim-core/releases
   - Release should be created automatically
   - Signed artifacts should be attached

### Post-Release Verification

```powershell
# Test installation in a fresh environment
python -m venv test_install
.\test_install\Scripts\Activate.ps1
pip install allooptim
python -c "from allooptim.optimizer.optimizer_list import get_all_optimizers; print(f'Found {len(get_all_optimizers())} optimizers')"
deactivate
Remove-Item -Recurse -Force test_install
```

## Troubleshooting

### "Trusted publisher not configured"
- Make sure you've added the pending publisher on PyPI/TestPyPI
- Project name must be exactly `allooptim` (lowercase)
- Repository must be `AlloOptim/allooptim-core`

### "Environment protection rules not satisfied"
- Make sure you've created the GitHub environments
- For `pypi`, ensure you've added required reviewers

### Workflow doesn't trigger
- Tag must start with `v` (e.g., `v0.3.0`)
- Make sure tag is pushed to GitHub: `git push origin --tags`

## Next Release

For the next release:

```powershell
# Update version in pyproject.toml
# Update CHANGELOG.md with new changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.3.1"
git push origin main
git tag -a v0.3.1 -m "Release v0.3.1"
git push origin v0.3.1
```

The workflow will handle everything automatically!
