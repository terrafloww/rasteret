# Contributing to Rasteret

Contributions are welcome: bug reports, feature ideas, docs improvements,
and code changes.

## Reporting issues

Open an issue on GitHub. Please include:

- What you tried and what happened
- Python version and `uv pip list` output
- Minimal reproducer if possible

For security vulnerabilities, use [private reporting](SECURITY.md) instead.

## Development setup

```bash
git clone https://github.com/terrafloww/rasteret.git
cd rasteret
uv sync --extra dev
uv run pre-commit install
uv run pytest -q
```

## Before submitting a PR

1. Run `uv run ruff check src/` and `uv run ruff format --check src/`
2. Run `uv run pytest -q` - all tests should pass
3. If you changed docs, run `uv run mkdocs build --strict`
4. Pre-commit hooks will run automatically on commit

## Adding a dataset to the catalog

Add a `DatasetDescriptor` to `src/rasteret/catalog.py`. Before opening a PR,
verify the [prerequisites](https://terrafloww.github.io/rasteret/how-to/dataset-catalog/#prerequisites-for-contributing-a-built-in-dataset):
STAC access works, band map points to parseable COGs, `build()` succeeds
end-to-end, and license metadata is sourced from the authoritative STAC API.

## Full contributor guide

Architecture overview, testing details, how to write ingest drivers, and
code style conventions are in the
[Contributing docs](https://terrafloww.github.io/rasteret/contributing/).

## License

By contributing, you agree that your contributions will be licensed under
the [Apache-2.0 License](LICENSE).
