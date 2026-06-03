# Contributing

Thank you for contributing to **head-orientations**!

## Pull Request Review Process

This repository uses a structured 3-step review workflow enforced via GitHub Actions and commit
status checks.

### How it works

1. **Open a PR** using the provided PR template.
2. **Three bot comments** are automatically posted to the PR — one per review step:
   - 🔵 **Code Review** — feature implementation, functionality, parameter names
   - 🧪 **Test Review** — implementation and completeness of tests
   - 📖 **Doc Review** — completeness, comprehensibility, concise and precise, examples
3. **Reviewers** from the team react with 👍 to each comment once they have completed that step.
4. Each step requires **2 unique reviewers** before its status check turns green.
5. A **maintainer** reviews the full PR, confirms all 3 status checks are ✅, and merges.

If a reviewer wants to revoke a sign-off, they remove their 👍 reaction. The status check reverts
to pending within ~10 minutes (the workflow polls on a schedule).

---

## Required Branch Protection Setup

> **Repository owner action required.** The following settings cannot be configured via workflow
> files and must be set manually in **Settings → Branches → Branch protection rules** for the
> `main` branch.

### Required status checks

Add these three required status checks so the merge button is blocked until all steps are complete:

- `review/code`
- `review/tests`
- `review/docs`

### Required approvals

- Enable **"Require a pull request before merging"**
- Set **"Required number of approvals before merging"** to `1` (the maintainer review)

### Recommended additional settings

- Enable **"Require status checks to pass before merging"**
- Enable **"Require branches to be up to date before merging"**
- Enable **"Do not allow bypassing the above settings"** for consistent enforcement
