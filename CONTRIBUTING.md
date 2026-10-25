# Contributing Protocol

To clone this project locally, issue

```shell
git clone https://github.com/BindsNET/bindsnet.git  # clones bindsnet repository
```

in the directory of your choice. This will place the repository's code in a directory titled `bindsnet`.

Install the project with [Poetry](https://python-poetry.org/) (version 2.4.3 is used in CI)

```shell
poetry install
poetry run pre-commit install
```


Now you can activate the project environment with `eval $(poetry env activate)` or run commands with `poetry run <command>`. For example, `poetry run python examples/mnist/conv_mnist.py`.

Please make sure the `Poetry` environment is activated when you commit your files! The `git commit` command will invoke `pre-commit`, which is installed with Poetry too. IDEs like PyCharm have plugins for `Poetry` and will activate the environment automatically.

Run the tests, they all should pass

```shell
poetry run pytest
```

Notable changes are recorded in [`CHANGELOG.md`](CHANGELOG.md); please add an entry to the
`Unreleased` section in your pull request.

## Tests that guard results

Some tests do not check one behaviour; they check that nothing changed. They fail
when a change alters simulated results or the public API, even if the change looks
harmless. Please do not edit them by hand to make them pass.

- **Golden-output tests (`test/regression/`).** Small seeded networks are run and their
  spikes, voltages, traces and weights are compared with stored reference files under
  `test/regression/`. Spikes must match exactly; other values must match to within
  1e-6. If your change alters these on purpose (for example, you fixed a formula), the
  reference files must be regenerated with the script in `test/regression/`, which
  refuses to run without `--reason "<why the results changed>"` and writes that reason
  into the files. Say in the pull request which files changed and why.
- **Public API snapshot (`test/api_surface.json`).** `test/test_api_surface.py` records
  the parameters and defaults of every public class, method and function in
  `bindsnet`. If you add, remove, rename or reorder a parameter, or change a default,
  regenerate it in the same pull request:

  ```shell
  poetry run python test/regen_api_surface.py --reason "<why the API changed>"
  ```

  The reason is required and is stored in the file. Removing or renaming a parameter
  can break other people's code, so expect to be asked why.
- **Every-class smoke tests (`test/network/test_all_classes.py`).** Every node,
  connection, learning rule, feature, monitor and model class is found automatically
  and run for a few steps. A new connection class needs an entry in `CONNECTION_SPECS`
  (its source and target shapes), a new model needs one in `MODEL_SPECS`, and a new
  learning rule needs one in `SUPPORTED`. Known bugs on `master` are listed in
  `KNOWN_BUGS` with their exact error; if your change fixes one, its test fails with
  "DID NOT RAISE", and you should remove the entry in the same pull request.
- **Operation counts (`test/regression/op_counts.json`).** `test/regression/test_op_counts.py`
  counts the tensor operations of one learning-rule update, one simulation step and one
  neuron update on small fixed networks. It fails when a count grows, because more
  operations per time step usually means slower code. If the extra work is needed,
  regenerate in the same pull request:

  ```shell
  poetry run python test/regression/regen_op_counts.py --reason "<why the hot path does more work>"
  ```

- **Speed comparison (informational).** The workflow `Speed comparison (informational)`
  (`.github/workflows/bench.yml`) times the benchmark workloads on the base and the head
  of your pull request, alternating ten rounds, and writes a table to the job summary
  (also downloadable as the `bench-ab` artifact). A workload is flagged SLOWER when the
  head takes more than 1.25 times as long (median over the rounds) and the lower end of
  the 95% bootstrap interval of that ratio is above 1. It is not a required check; shared CI machines are noisy, and a maintainer decides.

**Fixes that change results are released in 0.4.0, not in 0.3.x.** A correct fix that
changes what a simulation produces still changes published numbers for people using
0.3.x, so it waits for the next minor release. State in the pull request whether your
change alters results, and add a `CHANGELOG.md` entry that says so.

## Releasing to PyPI

1. Set `version` in `pyproject.toml` and `CITATION.cff`, and move the `Unreleased`
   entries in `CHANGELOG.md` under the new version.
2. Merge to `master`, then publish a GitHub Release whose tag equals that version.
3. `.github/workflows/publish.yml` builds the package, checks that the tag matches the
   `pyproject.toml` version, and uploads to PyPI through trusted publishing (no token).
   It can also be started by hand from the Actions tab.

A version number can be uploaded to PyPI only once, even after deletion.

All development should take place on a branch separate from master. To create a branch, issue

```shell
git branch [branch-name]  # create new branch
```

replacing `[branch-name]` with a simple and memorable name of choice; e.g., `git branch dan`. Switch to the newly created branch using

```shell
git checkout [branch-name]  # switch to a different branch of the repository
```

__Note__: Issue `git branch` with no arguments to list all branches currently being tracked, with an asterisk next to the currently used branch; e.g.,

```shell
$ git branch  # list all branches and indicate current branch
* dan
  devel
  hananel
  master
```

If new branches have been created on the remote repository, you may start tracking them with ```git pull --all```, and check them out using ```git checkout [branch-name]```, as before. ```git branch -a``` will list all locally tracked branches, and well as list all remote branches (which can be checked out!).

After making changes to the repository, issue a `git status` command to see which files have been modified. Then, use

```shell
git add [file-name(s) | -A]  # add modified or newly created files
```

to add one or more modified files (`file-name(s)`), or all modified files (`-A` or `--all`). These include newly created files. Issue

```shell
pre-commit run -a
```

to run the `pre-commit` tool that will automatically format your code with `black`. Issue

```shell
git commit -m "[commit-message]"  # Useful messages help when reverting / searching through history 
```

to "commit" your changes to your local repository, where `[commit-message]` is a _short yet descriptive_ note about what changes have been made.

Before pushing your changes to the remote repository, you must make sure that you have an up-to-date version of the `master` code. That is, if master has been updated while you have been making your changes, your code will be out of date with respect to the master branch. Issue

```shell
git pull  # gets all changes from remote repository
git merge master  # merges changes made in master branch with those made in your branch
```

and fix any merge conflicts that may have resulted, and re-commit after the fix with

```shell
git commit  # no -m message needed; merge messages are auto-generated
```

Push your changes back to the repository onto the same branch you are developing on. Issue

```shell
git push [origin] [branch-name]  # verbose; depends on push.default behavior settings
```

or,

```shell
git push  # concise; again, depends on push.default behavior
```

where `[origin]` is the name of the remote repository, and `[branch-name]` is the name of the branch you have developed on.

__Note__: See [push.default](https://git-scm.com/docs/git-config#git-config-pushdefault) for more information.

To merge your changes into the `master` branch (the definitive version of the project's code), open a pull request on the [webpage](https://github.com/BindsNET/bindsnet) of the project. You can select the `base` branch (typically `master`, to merge changes _into_ the definitive version of the code) and the `compare` branch (say, `dan`, if I added a new feature locally and want to add it to the project code). You may add an optional extended description of your pull request changes. If there are merge conflicts at this stage, you may fix these using GitHub's pull request review interface.

Assign reviewer(s) from the group of project contributors to perform a code review of your pull request. If the reviewer(s) are happy with your changes, you may then merge it in to the `master` branch. _Code review is crucial for the development of this project_, as the whole team should be held accountable for all changes.
