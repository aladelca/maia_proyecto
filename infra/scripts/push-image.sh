#!/bin/sh
set -eu
name=${1:?api, worker or training}
case "$name" in api|worker|training) ;; *) exit 2;; esac
region=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}
repository=$(terraform -chdir=infra/platform output -json repositories | python3 -c 'import json,sys;print(json.load(sys.stdin)[sys.argv[1]])' "$name")
registry=${repository%%/*}
revision=${CODEBUILD_RESOLVED_SOURCE_VERSION:-$(git rev-parse HEAD)}
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -n "$(git status --porcelain -- containers src services scripts/train_cloud.py)" ]; then revision="$revision-dirty"; fi
docker buildx build --build-arg "SOURCE_REVISION=$revision" --label "org.opencontainers.image.revision=$revision" --platform linux/amd64 --provenance=false --load -f "containers/Dockerfile.$name" -t "maia-$name:dev" .
tag=$(docker image inspect "maia-$name:dev" --format '{{.Id}}' | cut -c8-39)
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry" >/dev/null
if ! aws ecr describe-images --repository-name "${repository#*/}" --image-ids "imageTag=$tag" >/dev/null 2>&1; then
  docker tag "maia-$name:dev" "$repository:$tag"
  docker push "$repository:$tag"
fi
digest=$(aws ecr describe-images --repository-name "${repository#*/}" --image-ids "imageTag=$tag" --query 'imageDetails[0].imageDigest' --output text)
mkdir -p var/cloud
printf '%s@%s\n' "$repository" "$digest" > "var/cloud/$name-image.txt"
