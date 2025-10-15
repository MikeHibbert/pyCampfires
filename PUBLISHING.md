# Publishing Campfires to PyPI

This guide explains how to publish the Campfires package to PyPI.

## Prerequisites

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. Install required tools:
   ```bash
   pip install build twine
   ```

## Building the Package

1. Clean previous builds:
   ```bash
   rm -rf dist/ build/ *.egg-info/
   ```

2. Build the package:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/campfires-X.X.X-py3-none-any.whl` (wheel distribution)
   - `dist/campfires-X.X.X.tar.gz` (source distribution)

## Testing on TestPyPI

1. Upload to TestPyPI:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. Test installation from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ campfires
   ```

## Publishing to PyPI

1. Check the package:
   ```bash
   python -m twine check dist/*
   ```

2. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

3. Verify installation:
   ```bash
   pip install campfires
   ```

## Version Management

1. Update version in `pyproject.toml`:
   ```toml
   [project]
   version = "X.X.X"
   ```

2. Update version in `campfires/__init__.py`:
   ```python
   __version__ = "X.X.X"
   ```

3. Create a git tag:
   ```bash
   git tag vX.X.X
   git push origin vX.X.X
   ```

## Authentication

### Using API Tokens (Recommended)

1. Generate API tokens:
   - PyPI: Account settings → API tokens
   - TestPyPI: Account settings → API tokens

2. Configure credentials:
   ```bash
   # For PyPI
   python -m twine upload --username __token__ --password <your-token> dist/*
   
   # For TestPyPI
   python -m twine upload --repository testpypi --username __token__ --password <your-token> dist/*
   ```

### Using .pypirc (Alternative)

Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-testpypi-token>
```

## Automation

Consider using GitHub Actions for automated publishing:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Package Information

- **Package Name**: campfires
- **PyPI URL**: https://pypi.org/project/campfires/
- **Documentation**: Include in README.md
- **License**: MIT (included in package)

## Troubleshooting

1. **403 Forbidden**: Check API token permissions
2. **Package already exists**: Increment version number
3. **Invalid credentials**: Verify token/username
4. **File already exists**: Version already published, increment version

## Post-Publication

1. Test installation: `pip install campfires`
2. Update documentation with installation instructions
3. Announce the release
4. Monitor for issues and feedback