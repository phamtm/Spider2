# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Code Comments

- Prefer concise comments for each module, major class, and major function.
- Add comments for complicated flows, decisions, or constraints that are not obvious from the code.
- Keep comments easy to understand and avoid jargon when plain language works.
- Do not over-explain simple assignments or code that is already clear.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files
- This repo uses the shared local Dolt server on `127.0.0.1:3307` with database `sp`.
- Use normal `bd` commands against that shared server; do not start a second Dolt server or rely on `bd dolt pull` / `bd dolt push` for this workspace.

## Session Completion

**When ending a work session**, complete the local handoff steps below.

**Local workflow:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Commit local changes when appropriate**:
   ```bash
   git status
   git add <files>
   git commit -m "..."
   git status
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All intended changes are committed or clearly reported
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Do not push code or beads data unless explicitly requested
- Do not rely on autopush
- Keep work local by default
<!-- END BEADS INTEGRATION -->
