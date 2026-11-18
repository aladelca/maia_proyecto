#!/bin/sh
set -eu
stack=${1:?specify a persistent Terraform stack}
region=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}
account=$(aws sts get-caller-identity --query Account --output text)
terraform -chdir="infra/$stack" init -input=false -backend-config="bucket=maia-sleep-dev-$account-$region-state" -backend-config="region=$region" -backend-config="key=$stack/terraform.tfstate" -backend-config=use_lockfile=true -backend-config=encrypt=true
