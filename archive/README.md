# Archive

Earlier versions of this repository, kept for reference. The code in this
folder is not maintained and is superseded by the version at the repository
root. Cite the current release unless you need to reproduce an earlier one.

| Folder | Version | Snapshot | Release |
|---|---|---|---|
| `v4/` | 4.0 | `main` at commit `01e4631` (2026-07-20) | [v4.0.1](https://github.com/EAfrifa-Yamoah/rl-spatial-decision-making/releases/tag/v4.0.1) plus a later `CITATION.cff` fix and README update |

v5 was not published in this repository. The changes introduced in v6 are
listed in the root `CHANGELOG.md`.

Commands in `v4/README.md` assume `archive/v4/` is the working directory:

```bash
cd archive/v4
pip install -r requirements.txt
python -m pytest tests -q
python -m src.reproduce_all
```
