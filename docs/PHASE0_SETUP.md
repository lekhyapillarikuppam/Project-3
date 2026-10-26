# Phase 0 setup (Windows, PowerShell)

Goal: the repo runs on your laptop, tests pass, and the code is on your GitHub.
Time: about 30-45 minutes. Nothing here downloads the dataset or runs any model.

Commands marked `PS>` go in PowerShell (VS Code: Terminal > New Terminal).

## 1. Install the tools (skip any you already have)

| Tool | Get it | Check it |
| --- | --- | --- |
| Python 3.11 (3.10-3.13 all work) | python.org/downloads. Tick **Add python.exe to PATH** in the installer | `PS> python --version` |
| Git | git-scm.com/download/win (default options are fine) | `PS> git --version` |
| VS Code | code.visualstudio.com. Install the **Python** extension | open VS Code |
| Claude Code | Follow the install steps at docs.claude.com for Claude Code, then run `claude` once in a terminal to sign in | `PS> claude --version` |

Tell Git who you are (once per laptop):

```
PS> git config --global user.name "Your Name"
PS> git config --global user.email "you@example.com"
```

## 2. Unzip the repo

1. Download `ffs-evidence-inventory-phase0.zip`.
2. Right-click it, **Extract All**, to a short path such as `C:\Projects\`.
   You should now have `C:\Projects\ffs-evidence-inventory\`.
3. Open VS Code, **File > Open Folder**, choose that folder, click **Yes, I trust the authors**.
4. Open the terminal: **Terminal > New Terminal**. It starts in the repo folder.

The zip includes the git history. Check it:

```
PS> git log --oneline
```

You should see one commit.

## 3. Create the virtual environment and install packages

```
PS> python -m venv .venv
PS> .venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

If PowerShell says "running scripts is disabled", run this once and try again:

```
PS> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then install the core packages:

```
(.venv) PS> python -m pip install --upgrade pip
(.venv) PS> pip install -r requirements.txt
```

In VS Code press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and pick the one inside `.venv`.

Do **not** install `requirements-ml.txt` yet. That is Phase 1.

## 4. Check everything

```
(.venv) PS> python scripts/check_env.py
(.venv) PS> pytest
```

Expected:

- `check_env.py` ends with `RESULT: all good`. ML packages showing MISSING is fine today.
- `pytest` ends with `7 passed`.

If something fails, copy the full output and ask in this chat or in Claude Code.

## 5. Create the raw data folder

```
(.venv) PS> mkdir data\raw
```

Raw downloads go here. Git ignores everything in `data/`, so photos never get committed.

## 6. Put it on GitHub

1. On github.com click **New repository**. Name: `ffs-evidence-inventory`. Choose **Private**.
   Do **not** add a README, .gitignore or license (the repo already has them).
2. Copy the repository URL, then run:

```
(.venv) PS> git branch -M main
(.venv) PS> git remote add origin https://github.com/YOUR-USERNAME/ffs-evidence-inventory.git
(.venv) PS> git push -u origin main
```

GitHub will ask you to sign in the first time.

## 7. Try Claude Code in the repo

```
(.venv) PS> claude
```

Claude Code reads `CLAUDE.md` automatically. Try: `Summarize SPEC.md sections 1 to 3 in five lines.`
Then exit. Do not ask it to start Phase 1 yet.

## 8. Understand the spec (10 minutes)

Read `SPEC.md` sections 1-3. You should be able to answer:

- What is the research question?
- What is the difference between silver and gold labels?
- Why is the split by scene and not by photo?
- Why do we save all OWLv2 boxes to `owlv2_raw.csv`?

## Phase 0 gate

All of these must be true before Phase 1:

- [ ] `python scripts/check_env.py` says all good
- [ ] `pytest` passes
- [ ] The code is on your GitHub (`git status` says "up to date with origin/main")
- [ ] You have read `SPEC.md` sections 1-3
- [ ] `data\raw` exists

Then come back to this chat and say "Phase 0 done".
