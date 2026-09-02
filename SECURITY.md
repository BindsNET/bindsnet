# Security Policy

## Reporting a vulnerability

Please report security problems privately rather than in a public issue.

Use GitHub's private reporting form at
https://github.com/BindsNET/bindsnet/security/advisories/new, or email
hananel@hazan.org.il.

Please include what you found, how to reproduce it, and which version or commit
you were on. We aim to acknowledge reports within a few days.

## Supported versions

Security fixes are applied to the `master` branch and to the most recent release
on PyPI. Older releases are not patched.

## Repository integrity

BindsNET is a research library, and its git history is part of what users rely
on. The following controls are in place on this repository:

- Force-pushes and branch deletions are blocked on every branch.
- `master` requires a pull request with an approving review.

A supply-chain incident affecting this repository was reported and remediated in
September 2026; see issue #781 for the full account. No PyPI release was
affected. If you cloned this repository between 2026-08-29 and 2026-09-02 and
opened it in Visual Studio Code, please read that issue.

## What we will never do

BindsNET does not contain, and will never contain, code that runs automatically
when you open the project in an editor. There are no build hooks, no editor
tasks that execute on folder open, and no post-install scripts. If you find
anything of that shape in this repository, treat it as an incident and report it
using the process above.
