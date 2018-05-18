# Contributing Protocol

To clone this project locally, issue

```
git clone https://github.com/Hananel-Hazan/bindsnet.git  # clones bindsnet repository
```

in the directory of your choice. This will place the repository's code in a directory titled `bindsnet`.

All development should take place on a branch separate from master. To create a branch, issue

```
git branch [branch-name]  # create new branch
```

replacing `[branch-name]` with a simple and memorable name of choice; e.g., `git branch dan`. Switch to the newly created branch using

```
git checkout [branch-name]  # switch to a different branch of the repository
```

__Note__: Issue `git branch` with no arguments to list all branches currently being tracked, with an asterisk next to the currently used branch; e.g.,

```
$ git branch  # list all branches and indicate current branch
* dan
  devel
  hananel
  master
```

If new branches have been created on the remote repository, you may start tracking them with ```git pull --all```, and check them out using ```git checkout [branch-name]```, as before. ```git branch -a``` will list all locally tracked branches, and well as list all remote branches (which can be checked out!).

After making changes to the repository, issue a `git status` command to see which files have been modified. Then, use

```
git add [file-name(s) | -A]  # add modified or newly created files
```

to add one or more modified files (`file-name(s)`), or all modified files (`-A` or `--all`). These include newly created files. Issue

```
git commit -m "[commit-message]"  # Useful messages help when reverting / searching through history 
```

to "commit" your changes to your local repository, where `[commit-message]` is a _short yet descriptive_ note about what changes have been made.

Before pushing your changes to the remote repository, you must make sure that you have an up-to-date version of the `master` code. That is, if master has been updated while you have been making your changes, your code will be out of date with respect to the master branch. Issue

```
git pull  # gets all changes from remote repository
git merge master  # merges changes made in master branch with those made in your branch
```

and fix any merge conflicts that may have resulted, and re-commit after the fix with

```
git commit  # no -m message needed; merge messages are auto-generated
```

Push your changes back to the repository onto the same branch you are developing on. Issue

```
git push [origin] [branch-name]  # verbose; depends on push.default behavior settings
```

or,

```
git push  # concise; again, depends on push.default behavior
```

where `[origin]` is the name of the remote repository, and `[branch-name]` is the name of the branch you have developed on.

__Note__: See [push.default](https://git-scm.com/docs/git-config#git-config-pushdefault) for more information.

To merge your changes into the `master` branch (the definitive version of the project's code), open a pull request on the [webpage](https://github.com/Hananel-Hazan/bindsnet) of the project. You can select the `base` branch (typically `master`, to merge changes _into_ the definitive version of the code) and the `compare` branch (say, `dan`, if I added a new feature locally and want to add it to the project code). You may add an optional extended description of your pull request changes. If there are merge conflicts at this stage, you may fix these using GitHub's pull request review interface.

Assign reviewer(s) from the group of project contributors to perform a code review of your pull request. If the reviewer(s) are happy with your changes, you may then merge it in to the `master` branch. _Code review is crucial for the development of this project_, as the whole team should be held accountable for all changes.
