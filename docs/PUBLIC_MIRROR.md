# Public Repo Workflow

This repository uses a two-repo workflow:

- local/private source repo: full working history, `research_pack/`, and local
  tooling
- local/public mirror repo: publishable subset only

## Mirrored Subset

The sync script copies only:

- `README.md`
- `requirements.txt`
- `docs/`
- `spec/Spec.pdf`
- `spec/overview.md`
- `spec/contracts.md`
- `spec/environment.md`
- `spec/parity.md`
- `spec/acceptance-tests.md`
- `spec/plan.md`
- `.github/workflows/gf01-gate.yml`
- `gf01/`
- `tests/`

Everything else stays private by default.

## One-Command Sync

From the source repo root:

```bash
python3 scripts/sync_public_repo.py --init-git --commit
```

This will:

1. refresh `public_repo/` from the explicit allowlist,
2. initialize `public_repo/.git` if needed,
3. commit mirror changes in `public_repo`.

## Push To GitHub

After creating the mirror remote:

```bash
cd public_repo
git remote add origin <REMOTE_URL>
git push -u origin main
```

For subsequent updates:

```bash
cd ..
python3 scripts/sync_public_repo.py --commit
cd public_repo
git push
```

## Safety Rules

- Keep the allowlist explicit; do not mirror `research_pack/`, local scripts, or
  private artifact directories by accident.
- Keep `requirements.txt` limited to dependencies needed by mirrored/public
  files.
- Treat local duplicate artifacts such as `foo 2.py`, `bar 2.md`, or
  `commands 2/` as hygiene failures. They are not part of the mirror workflow,
  should not be ignored, and should be deleted after confirming they add no new
  information.
- Keep the GitHub branch protection rule requiring both
  `GF01 Gate / gate` and `GF01 Gate / release-candidate` on the mirror
  `main` branch.
