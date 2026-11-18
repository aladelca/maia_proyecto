#!/bin/bash
# Isolate each campaign's disposable instance from the long-lived serving and training stacks.
set -euo pipefail
mode=${1:-refit}
case "$mode" in refit|tune) ;; *) exit 2;; esac
job=${CAMPAIGN_ID:-$(python3 -c 'import uuid;print(uuid.uuid4())')}
python3 -c 'import sys,uuid;assert str(uuid.UUID(sys.argv[1]))==sys.argv[1]' "$job"
attempt=$(python3 -c 'import uuid;print(uuid.uuid4())')
root="$(pwd)/var/cloud/fit/$job"
mkdir -p "$root"
terraform -chdir=infra/training output -json runner >"$root/runner.json"
python3 - "$root" "$job" "$attempt" "$mode" <<'PY'
from pathlib import Path
import json,sys,os
folder=Path(sys.argv[1])
runner=json.loads((folder/"runner.json").read_text())
(folder/"fit.tfvars.json").write_text(json.dumps(dict(runner=runner,run_id=sys.argv[2],attempt=sys.argv[3],mode=sys.argv[4],instance_type=os.environ.get("TRAIN_INSTANCE_TYPE","t3.large"))))
PY
region=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}
account=$(aws sts get-caller-identity --query Account --output text)
bucket=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["inputs"]["buckets"]["research"])' "$root/runner.json")
export TF_DATA_DIR="$root/terraform"
tf() { terraform -chdir=infra/fit "$@"; }
tf init -input=false -backend-config="bucket=maia-sleep-dev-$account-$region-state" -backend-config="key=fit/$job.tfstate" -backend-config="region=$region" -backend-config=use_lockfile=true -backend-config=encrypt=true
# A live instance owns this campaign until it finishes. Each new submission gets its own UUID.
existing=$(tf output -json | python3 -c 'import json,sys;print(json.load(sys.stdin).get("instance_id",{}).get("value",""))')
if [[ -n "$existing" ]]; then
  state=$(aws ec2 describe-instances --instance-ids "$existing" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" != terminated ]]; then echo "Campaign already has instance $existing ($state)" >&2;exit 1;fi
fi
tf plan -input=false -var="region=$region" -var-file="$root/fit.tfvars.json" -out="$root/fit.tfplan"
tf apply -input=false "$root/fit.tfplan"
id=$(tf output -raw instance_id)
echo "Campaign $job on $id. Logs: /maia/sleep/dev/training"
key="candidates/$job/attempts/$attempt/exit.json"
while ! aws s3api get-object --bucket "$bucket" --key "$key" "$root/exit.json" >/dev/null 2>"$root/poll-error.log"; do
  state=$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" == terminated ]]; then echo "Instance terminated before recording exit status" >&2; exit 1; fi
  sleep 15
done
tf destroy -input=false -auto-approve -var="region=$region" -var-file="$root/fit.tfvars.json"
python3 -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["exit_code"])' "$root/exit.json"
