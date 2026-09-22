# Repository Instructions for Codex

These instructions apply to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## General working style

- Work from the current project files and the user's stated task.
- Prefer the smallest set of inspections and commands needed to complete the
  task.
- Reuse information already obtained during the current task. Do not repeat a
  search, file read, test, or status check unless a file has changed or the
  previous output was incomplete.
- Batch independent read-only checks into one command when this keeps the
  output readable. Do not issue several nearly identical commands.
- Keep command output focused. Use explicit paths, bounded line ranges, and
  targeted patterns.

## Superpowers usage

Superpowers skills may be used for planning, decomposition,
verification, and code review.

However, this repository is primarily a machine-learning research project,
not a conventional product software project.

For research experiments:

- Do not mechanically apply test-driven development when there is no
  meaningful failing unit test to write.
- Do not create git commits, branches, or worktrees unless explicitly
  requested by the user.
- Do not treat experimental metric changes as software test failures.
- Prefer reproducibility checks, data-integrity checks, pipeline checks,
  and result validation over artificial unit tests.
- Plans should preserve the experimental methodology and scientific
  comparison design.

## Git ownership and restrictions

The user owns all Git history and remote operations.

- Do not run `git add`, `git commit`, `git push`, `git pull`, `git fetch`,
  `git merge`, `git rebase`, `git cherry-pick`, `git tag`, branch creation or
  deletion, or any other Git command that changes local history, branches, the
  index, or a remote repository.
- Do not run destructive Git commands such as `git reset`, `git restore`,
  `git checkout --`, `git clean`, or force operations.
- Do not create commits or upload changes. The user will perform all Git
  staging, commits, synchronization, and uploads.
- The agent may inspect a focused diff of files it modified during the current
  task when needed to verify its own changes.
- Do not run general repository-wide Git status or diff commands merely for
  curiosity or routine reporting.
- Inspect Git status, diffs, logs, trees, or historical versions only when the
  current task genuinely requires repository history or a comparison with a
  committed version. Before doing so, briefly tell the user which Git command
  is needed and what question it will answer.
- Do not run multiple overlapping Git inspections such as `git status`,
  `git diff --stat`, and `git diff --name-status` by default. Select the single
  command that answers the current question. Run another only if the first
  result is insufficient.

## Search and file-reading policy

- Prefer direct reads of known files over repository-wide searches.
- Use `rg` only when the location of relevant code or text is unknown, or when
  checking references across files is necessary.
- Make `rg` searches targeted: specify the exact pattern, directory, and file
  types. Exclude generated folders such as `.git`, `.venv`, `__pycache__`,
  build outputs, and large result directories unless they are directly
  relevant.
- Combine related patterns or explicit file paths into one `rg` call when the
  resulting output remains understandable.
- Do not use repeated broad searches with slightly different patterns.
- Do not use `rg -n "^"` merely to print an entire known file. Read the file
  directly, and request only the relevant line range when practical.
- Do not scan CSV result files, generated artifacts, Markdown, JSON, and Python
  files together unless the task specifically requires a cross-format
  reference check.
- If a broad repository-wide search is necessary, explain its purpose before
  running it.

## Read-only commands

- Read-only inspection commands may be executed directly without asking the
  user when the environment permits them.
- Examples include targeted uses of `Get-Content`, `Get-ChildItem`,
  `Select-String`, `Import-Csv`, `Format-List`, `Format-Table`, `rg`, file
  listing, and bounded source inspection.
- Even for read-only commands, follow the search limits above and avoid
  redundant or excessively broad inspection.
- Reading a file does not authorize rewriting, moving, renaming, or deleting
  it.

## Commands that require approval or explanation

- If a command requires user approval and is not already covered as a direct,
  read-only inspection, explain it before requesting approval.
- The explanation must state:
  1. the exact command or a faithful concise form of it;
  2. what it will read, create, modify, install, execute, or delete;
  3. why it is needed for the current task;
  4. the expected scope and any meaningful risk or side effect.
- Do not describe a command only as "needed for verification." State the
  concrete property it verifies.
- Do not bundle unrelated commands into one approval request.
- If a safer read-only or narrower command can answer the same question, use
  that command instead.

## Tests and Python execution

- Run the smallest relevant test target first, such as one test module or one
  affected test class.
- Distinguish software verification from scientific experiment execution.
  A quick import check, dry run, or reduced-data sanity check may be used to
  verify code without rerunning the full experiment.
- Run the complete test suite only when the change affects shared code, when a
  final regression check is needed, or when the user explicitly requests it.
- Do not repeatedly run the same unchanged test suite.
- Before running a long, resource-intensive, data-generating, or model-training
  command, explain what will run, which files it may write, and why it is
  necessary.
- Do not rerun completed machine-learning experiments merely to inspect or
  explain existing results.

## File changes

- Make only changes required by the user's request and preserve unrelated user
  work.
- Prefer patch-based edits for source files.
- Do not delete, rename, move, or overwrite material files unless the user has
  requested that action or it is an explicit part of the approved task.
- After editing, report the files changed and the focused verification
  performed. Do not perform Git staging, commits, or uploads.

