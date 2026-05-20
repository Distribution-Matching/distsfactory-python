# Publishing

This package publishes to PyPI via GitHub Actions using [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no API tokens stored in the repo).

## One-time setup

Both steps must happen *before* the first tag is pushed.

### 1. PyPI: configure trusted publisher

Create a stub project on PyPI (or use the "pending publisher" flow) and add a trusted publisher:

- **PyPI Project Name**: `distsfactory`
- **Owner**: `Distribution-Matching`
- **Repository**: `distsfactory-python`
- **Workflow**: `publish.yml`
- **Environment**: `pypi`

Repeat on [test.pypi.org](https://test.pypi.org/) with environment `testpypi` if you want to dry-run via `workflow_dispatch`.

### 2. GitHub: create the two deployment environments

`Settings → Environments → New environment`:

- `pypi` (production) — optionally require manual approval
- `testpypi` (dry-runs)

No secrets needed; OIDC handles auth.

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml and commit
vim pyproject.toml                     # change version = "X.Y.Z"
git commit -am "Release vX.Y.Z"
git push

# 2. Tag and push
git tag vX.Y.Z
git push origin vX.Y.Z
```

The `publish` workflow then:

1. Verifies `vX.Y.Z` matches `pyproject.toml`'s version (fails fast if not).
2. Builds the sdist + wheel and runs `twine check`.
3. Uploads to PyPI via the `pypi` environment.

After it lands, create a matching GitHub release pointing at the tag with a copy of the changelog entry.

## Dry-run to TestPyPI

Trigger the workflow manually (`Actions → publish → Run workflow`) on `main`. It builds and uploads to TestPyPI only — production PyPI publishing is tag-gated.

Verify with:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ distsfactory
```

## Version bump strategy

- **Patch (0.1.x)** — bug fixes, internal refactors, doc fixes.
- **Minor (0.x.0)** — new families, new spec types, parity additions.
- **Major (x.0.0)** — breaking API changes. Out of scope until 1.0.

The single source of truth for the installed version is `pyproject.toml`; `__init__.py` reads it via `importlib.metadata`.
