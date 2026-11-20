# Contributions

We warmly welcome contributions to ArcaNN. If you have ideas, code contributions, or suggested optimizations, please feel free to submit them.

## Code layout

The `arcann_training/` package is organized by **step**, mirroring the iterative procedure: `initialization/`, `exploration/`, `labeling/`, `training/`, and `test/` each contain one module per *phase* (`prepare.py`, `launch.py`, `check.py`, ...), plus a `utils.py` for step-specific helpers. Code shared across steps (JSON/XML/YAML handling, machine and Slurm configuration, LAMMPS/PLUMED file helpers, logging) lives in `common/`. Unit tests live in `arcann_training/unittests/`.

## Development setup

The project is managed with [`uv`](https://docs.astral.sh/uv/) (the dev/lint/doc tooling is declared as `dependency-groups` in `pyproject.toml`, not as `pip` extras). From a clone of the repository:

```bash
uv sync --locked --dev
pre-commit install
```

This creates a local `.venv` with ArcaNN installed in editable mode plus the `dev` group (`ruff`, `pre-commit`). If you'd rather use a plain `pip`/Conda environment (e.g. the one from [Installation](../getting-started/installation.md)), install it in editable mode and add `ruff` yourself: `pip install -e . && pip install ruff`.

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, and Python's built-in `unittest` for tests; both are wired into `.pre-commit-config.yaml` and run automatically on `git commit`. You can also run them manually:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run python -m unittest discover arcann_training/unittests
```

CI (see `.github/workflows/`) runs this same unit test matrix (Python 3.10–3.13) and builds the documentation site, so make sure both pass locally before opening a pull request. You can preview the docs locally with `uv sync --group docs && uv run mkdocs serve`.

The docs are deployed to GitHub Pages on every push to `main` (and, for now, `develop`). Every pull request targeting `main` or `develop` from a branch of this repository also gets a preview. It is published under `pr-preview/pr-<N>/`, and a bot comment on the PR links to it. The preview is removed when the PR is closed. Pull requests from forks are built but get no preview.

## Commit messages and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat: ...`, `fix: ...`, `docs: ...`, `feat!: ...` for breaking changes); a `commit-msg` hook installed by `pre-commit install` checks this. Versions follow [semantic versioning](https://semver.org/) and only change when a maintainer cuts a release, not on every commit or pull request.

To release (maintainers):

1. Merge everything that should be in the release into `main`.
2. In GitHub, go to **Actions → Release PR → Run workflow** (on `main`). Leave *increment* on `auto` to derive it from the commit messages, or force `PATCH`/`MINOR`/`MAJOR`; pick a *prerelease* phase (`alpha`, `beta`, `rc`) for a pre-release.
3. The workflow runs `cz bump` on a `release/vX.Y.Z` branch (version in `pyproject.toml` and `CITATION.cff`, new `CHANGELOG.md` section) and opens a pull request to `main`. Its checks only start after a maintainer clicks **Approve and run** on the PR (GitHub holds checks on pull requests opened by GitHub Actions). Review and merge it.
4. Merging runs `.github/workflows/publish.yml`: it tags `vX.Y.Z` on the merge commit, runs the tests, publishes to [PyPI](https://pypi.org/project/arcann_training/) (with build attestations) and creates the GitHub Release, which triggers the Zenodo archive and its DOI. It then opens a pull request from a `back-merge/vX.Y.Z` branch (made from `develop`, with the release merged in) to `develop`; approve its checks and merge it, so the next `develop` → `main` pull request is up to date.

Pre-releases (`alpha`, `beta`, `rc`) go through the same pipeline: they are published on PyPI as pre-releases (installed only with `pip install --pre`), marked "Pre-release" on GitHub, deployed as a docs version that does not become `latest`, and **also archived on Zenodo** with their own permanent DOI. Until the final release is out, the Zenodo concept DOI resolves to the newest pre-release, so use pre-releases when testers need an installable version, not for every fix.

A release PR can also be prepared by hand; the result is the same:

```bash
git switch main && git pull
git switch -c release/vX.Y.Z           # the branch name must start with release/
uv run cz bump                         # optionally --increment MAJOR|MINOR|PATCH
git tag -d vX.Y.Z                      # cz bump also tags locally: never push that tag
git push origin release/vX.Y.Z
gh pr create --base main --title "chore(release): vX.Y.Z"
```

If the current version was never released (its `publish.yml` run failed before tagging, e.g. PyPI rejected the upload), `cz bump` stops with "No tag found to do an incremental changelog". The *Release PR* workflow handles this itself; by hand, first add a temporary local tag on the commit that set that version, `git tag vX.Y.Z "$(git log -1 --format=%H -G '^version = "X.Y.Z"$' -- pyproject.toml)"`, and delete it after the bump without pushing it.

The tag must be created by `publish.yml` on the merge commit: if `vX.Y.Z` already exists on GitHub, merging publishes nothing (the *Version guard* check fails in that case).

To preview the next version locally: `uv run cz bump --dry-run`. Never change `version` in `pyproject.toml` by hand: any new, untagged version that reaches `main` is published. The *Version guard* check fails any pull request that changes the version, unless it comes from a `release/*` branch, or from a `back-merge/*` branch or `main` (bringing a release back into `develop`).

## Authorship and credit

Every contribution is credited, but not every contribution makes someone an author of the software citation. ArcaNN keeps two lists:

- **Authors** are listed in [`CITATION.cff`](https://github.com/champagnn/arcann_training/blob/main/CITATION.cff), which GitHub ("Cite this repository") and Zenodo use for the software record and its DOI. Authorship is for substantial contributions to the code, its design or its scientific methodology, for example a new feature, interface or workflow step, a significant refactor, sustained maintenance, or scientific supervision of such work.
- **Contributors** are listed in [`AUTHORS.md`](https://github.com/champagnn/arcann_training/blob/main/AUTHORS.md). This covers smaller code fixes, documentation, examples, testing, bug reports and reviews. `AUTHORS.md` also lists the authors and is the full record of everyone who has helped.

How it works in practice:

- Add yourself to the Contributors section of `AUTHORS.md` in your first pull request.
- If you think your work qualifies for authorship, say so in the pull request or open an issue. You can also be proposed by someone else. The maintainers decide, and authorship takes effect from the next release.
- Authors give their name, ORCID and affiliation in `CITATION.cff`. Use the affiliation(s) under which you did the work; if you have more than one, separate them with semicolons.
- Author order is decided by the maintainers and reflects the size of each contribution. New authors are normally added after the existing ones.
- Authors are never removed from `CITATION.cff` because they stopped contributing: every released version credits the people whose work it contains.

Authorship of the software record is separate from authorship of papers about ArcaNN, which follows the usual rules of the journal and the people involved.

## Reporting bugs

See [Unexpected Behavior](./unexpected_behavior.md) for how to report issues.
