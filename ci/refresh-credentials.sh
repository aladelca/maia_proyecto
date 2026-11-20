#!/bin/sh
# Copy the current AWS Academy session from .env into the GitHub environment used by deploy.yml.
set -eu
environment=${1:-dev}
set -a
. ./.env
set +a
aws sts get-caller-identity --query Arn --output text >/dev/null
for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
  value=$(printenv "$name") || { echo "$name is missing from .env" >&2; exit 1; }
  printf '%s' "$value" | gh secret set "$name" --env "$environment"
done
