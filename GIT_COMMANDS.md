# Commonly Used Git Commands

This document provides a quick reference for frequently used Git commands.

## 1. Basic Setup & Configuration

*   `git init`
    *   Initializes a new Git repository in the current directory.
    *   Creates a hidden `.git` folder.

*   `git config --global user.name "Your Name"`
    *   Sets your global Git username.

*   `git config --global user.email "your_email@example.com"`
    *   Sets your global Git email address.

*   `git config --list`
    *   Lists all Git configurations.

## 2. Staging & Committing Changes

*   `git status`
    *   Shows the status of changes as untracked, modified, or staged.

*   `git add <file>`
    *   Stages a specific file for the next commit.

*   `git add .`
    *   Stages all new and modified files in the current directory and its subdirectories.

*   `git reset <file>`
    *   Unstages a specific file (moves it from staging area back to modified state).

*   `git commit -m "Your commit message"`
    *   Records the staged changes as a new commit with a descriptive message.

*   `git commit --amend`
    *   Amends the last commit. Useful for correcting commit messages or adding forgotten changes to the last commit (before pushing).

## 3. Branching & Merging

*   `git branch`
    *   Lists all local branches.

*   `git branch <branch-name>`
    *   Creates a new branch.

*   `git checkout <branch-name>`
    *   Switches to an existing branch.

*   `git checkout -b <new-branch-name>`
    *   Creates a new branch and switches to it immediately.

*   `git merge <branch-name>`
    *   Merges the specified branch into the current branch.

*   `git branch -d <branch-name>`
    *   Deletes a local branch (only if it has been merged).

*   `git branch -D <branch-name>`
    *   Force deletes a local branch (even if it hasn't been merged).

## 4. Remote Repositories

*   `git remote add origin <repository-url>`
    *   Adds a remote repository (usually named `origin`) to your local repository.

*   `git remote -v`
    *   Lists all configured remote repositories.

*   `git push -u origin <branch-name>`
    *   Pushes your local branch to the remote repository for the first time, setting it up to track the remote branch.

*   `git push`
    *   Pushes your local changes to the remote repository (after the initial `-u` push).

*   `git pull`
    *   Fetches changes from the remote repository and merges them into your current local branch.

*   `git clone <repository-url>`
    *   Clones an entire repository from a remote URL to your local machine.

## 5. Viewing History

*   `git log`
    *   Shows the commit history.

*   `git log --oneline`
    *   Shows a condensed commit history.

*   `git diff`
    *   Shows changes between the working directory and the staging area.

*   `git diff --staged`
    *   Shows changes between the staging area and the last commit.

*   `git diff <commit1> <commit2>`
    *   Shows changes between two specific commits.

## 6. Undoing Changes

*   `git restore <file>`
    *   Discards changes in the working directory for a specific file (since the last commit or staging).

*   `git restore .`
    *   Discards all changes in the working directory.

*   `git revert <commit-hash>`
    *   Creates a new commit that undoes the changes made in a previous commit (safe for pushed commits).

*   `git reset --hard <commit-hash>`
    *   Resets the current branch to a specific commit, discarding all subsequent changes (DANGEROUS, use with caution, especially on pushed commits).

---
