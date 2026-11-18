#!/bin/sh
set -eu
region=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}
account=$(aws sts get-caller-identity --query Account --output text)
bucket=maia-sleep-dev-$account-$region-state
remote=infra/bootstrap/backend.auto.tf.json
if [ ! -f "$remote" ] && ! aws s3api head-object --bucket "$bucket" --key bootstrap/terraform.tfstate >/dev/null 2>&1; then
  terraform -chdir=infra/bootstrap init -input=false
  terraform -chdir=infra/bootstrap apply -input=false -auto-approve -var="region=$region"
fi
printf '%s\n' '{"terraform":{"backend":{"s3":{}}}}' > "$remote"
terraform -chdir=infra/bootstrap init -input=false -force-copy -backend-config="bucket=$bucket" -backend-config="key=bootstrap/terraform.tfstate" -backend-config="region=$region" -backend-config=encrypt=true -backend-config=use_lockfile=true
terraform -chdir=infra/bootstrap apply -input=false -auto-approve -var="region=$region"
