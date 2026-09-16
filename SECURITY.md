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

## Loading saved networks and models

Saved network files are Python pickle files. Loading one runs whatever code is
stored inside it. This means:

**Only load network or model files you created yourself, or that came from a
source you trust.** A file from a download, a model-sharing site, shared group
storage, or a collaborator can run arbitrary commands on your machine as soon
as you load it. There is no way to inspect a pickle file safely first.

This affects the following functions, all of which load a file path you give
them:

- `bindsnet.network.load`
- `bindsnet.conversion.ann_to_snn`, when passed a path instead of a module
- `bindsnet.conversion.data_based_normalization`, when passed a path instead of
  a module

This is the standard behaviour of `torch.load`, and it applies to saved models
across the PyTorch ecosystem rather than being specific to BindsNET. We do not
treat it as a vulnerability in BindsNET, because there is no way to load a
saved network object without it. We do treat it as something you need to be
told about clearly, which is what this section and the warnings in those
functions' documentation are for.

`bindsnet.network.load` accepts `weights_only=True`, which asks PyTorch to
refuse to execute code while loading. It cannot read files written by
`Network.save`, because those store the whole network object rather than a
plain tensor state dictionary, so it is only useful for files you know contain
plain tensors.

If you need to share a trained network with people who cannot verify where it
came from, share the weights as a tensor state dictionary rather than a pickled
network object.

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
