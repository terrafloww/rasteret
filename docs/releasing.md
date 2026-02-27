# Releasing

Maintainer notes for cutting a new release.

## Checklist

### 1. Version bump

- [ ] Version is derived from git tags via `hatch-vcs` (no manual edit in `pyproject.toml`)
- [ ] Update `version` and `date-released` in `CITATION.cff`

### 2. Changelog

- [ ] Add a new `## vX.Y.Z` section to `docs/changelog.md`
- [ ] Summarize highlights, stability changes, and breaking changes

### 3. Lint and format

```bash
uv run ruff check src/ docs/
uv run ruff format --check src/
```

### 4. Unit tests

```bash
uv sync --extra dev --extra torchgeo
uv run pytest -q
```

### 5. Live smoke tests (network)

Network tests are excluded by default. Run them locally when you're about
to cut a release:

```bash
uv sync --extra dev --extra azure
uv run pytest -m network -q
```

To require that **all** live checks run (fail if any network test is skipped):

```bash
uv run pytest -m network -q --network-strict
```

### 6. Docs build

```bash
uv run mkdocs build --strict
```

Check for warnings and broken links.

### 7. Build artifacts

```bash
uv build
```

Verify both sdist and wheel are produced.

### 8. Tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 9. PyPI publish (when applicable)

```bash
uv publish
```
