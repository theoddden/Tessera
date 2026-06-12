#!/bin/bash
set -e

TOKEN="${GITHUB_TOKEN:?GITHUB_TOKEN environment variable required}"
REPO_NAME="Tessera"
USERNAME="theoddden"

# Create repo
curl -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"Tessera: LoRA adapter generation and composition system\",\"private\":false}"

# Configure git remote
git remote set-url origin https://${USERNAME}:${TOKEN}@github.com/${USERNAME}/${REPO_NAME}.git

# Push
git push -u origin main
