# Release Workflow Setup Guide

## 🚀 Quick Start

This repository uses **GitHub Actions + PyPI Trusted Publishers** for automated, secure package releases.

## 📋 One-Time Setup

### 1. Configure PyPI Trusted Publishers

#### TestPyPI (for testing releases)
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new pending publisher with these details:
   - **PyPI Project Name**: `allooptim`
   - **Owner**: `AlloOptim`
   - **Repository name**: `allooptim-core`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `testpypi`

#### PyPI (for production releases)
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher with these details:
   - **PyPI Project Name**: `allooptim`
   - **Owner**: `AlloOptim`
   - **Repository name**: `allooptim-core`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

### 2. Configure GitHub Environments

In your GitHub repository settings → Environments:

#### Create `testpypi` environment:
- No protection rules needed (for automatic testing)

#### Create `pypi` environment:
- ✅ **Required reviewers**: Add maintainers who should approve releases
- ✅ **Wait timer**: Optional (e.g., 5 minutes)
- This adds a manual approval gate before production releases

## 🎯 How to Release

### Standard Release Process

1. **Update version** in `pyproject.toml`:
   ```toml
   [tool.poetry]
   version = "0.3.1"
   ```

2. **Commit and tag**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.3.1"
   git tag v0.3.1
   git push origin main --tags
   ```

3. **Workflow automatically**:
   - Builds the package
   - Publishes to TestPyPI
   - Waits for manual approval for PyPI
   - Publishes to PyPI
   - Creates GitHub Release with signed artifacts

### Release Checklist

- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run all tests locally: `poetry run pytest tests/`
- [ ] Update version in `pyproject.toml`
- [ ] Create git tag: `git tag v0.x.y`
- [ ] Push tag: `git push --tags`
- [ ] Monitor GitHub Actions workflow
- [ ] Approve PyPI deployment when prompted
- [ ] Verify package on PyPI
- [ ] Update GitHub Release notes if needed

## 🔒 Security Features

- ✅ **No API tokens** stored in GitHub secrets
- ✅ **Short-lived OIDC tokens** (15-minute expiry)
- ✅ **Sigstore attestations** for supply chain security
- ✅ **Manual approval** required for production releases
- ✅ **Automatic signature** verification

## 🛠️ Workflow Files

- `.github/workflows/publish.yml` - Main release workflow

## 📚 Resources

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python Packages](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
