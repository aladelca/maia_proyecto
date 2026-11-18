#!/bin/sh
set -eu
version=1.15.8
archive="terraform_${version}_linux_amd64.zip"
directory=$(mktemp -d)
cd "$directory"
curl -fsSLO "https://releases.hashicorp.com/terraform/$version/$archive"
curl -fsSLO "https://releases.hashicorp.com/terraform/$version/terraform_${version}_SHA256SUMS"
awk -v archive="$archive" '$2==archive {print; found=1} END {if(!found) exit 1}' "terraform_${version}_SHA256SUMS" | sha256sum -c -
unzip -q "$archive" terraform
install -m 0755 terraform /usr/local/bin/terraform
