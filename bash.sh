#!/bin/bash
set -e

# Number of commits to consider (default 187; adjust if needed)
NUM_COMMITS=${1:-187}

# Ensure your git config is correct
git config user.name "Bruno Magalhaes"
git config user.email "bruno.magalhaes@synthesia.io"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo "Starting interactive rebase of the last $NUM_COMMITS commits with sign-off automation..."

# Run interactive rebase with a one-line exec command.
# This command checks if the commit’s author email is yours and if its message lacks a Signed-off-by line,
# it amends the commit (with -s) to add the sign-off.
git rebase -i HEAD~${NUM_COMMITS} --exec 'if [ "$(git show -s --format=%ae HEAD)" = "bruno.magalhaes@synthesia.io" ]; then if ! git log -1 --pretty=%B | grep -q "Signed-off-by:"; then echo "Adding Signed-off-by for commit $(git rev-parse HEAD)"; git commit --amend --no-edit -s; fi; fi'

# If a conflict occurs during the rebase, resolve it automatically.
# This loop will continue until the rebase completes.
while [ -d .git/rebase-apply ] || [ -d .git/rebase-merge ]; do
  echo "Conflict detected, automatically resolving by choosing current changes."
  # Checkout the current version of all files (i.e., "ours")
  git checkout --ours .
  # Stage the resolved changes
  git add -A
  # Continue the rebase
  git rebase --continue
done

echo "Force pushing the updated branch to origin/$CURRENT_BRANCH..."
#git push --force-with-lease origin "$CURRENT_BRANCH"
echo "DCO remediation complete for your commits."

