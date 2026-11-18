terraform {
  required_version = ">= 1.10, < 2.0"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
provider "aws" { region = var.region }
data "aws_caller_identity" "current" {}
locals {
  name    = "maia-sleep-dev"
  storage = toset(["research", "releases", "jobs"])
}
resource "terraform_data" "buckets" {
  for_each = local.storage
  input    = "${local.name}-${data.aws_caller_identity.current.account_id}-${var.region}-${each.key}"
  provisioner "local-exec" { command = "python3 ../scripts/ensure_bucket.py ${self.input} ${var.region}" }
}
resource "aws_s3_bucket_versioning" "private" {
  for_each = terraform_data.buckets
  bucket   = each.value.output
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_public_access_block" "private" {
  for_each                = terraform_data.buckets
  bucket                  = each.value.output
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_lifecycle_configuration" "jobs" {
  bucket     = terraform_data.buckets["jobs"].output
  depends_on = [aws_s3_bucket_versioning.private]
  rule {
    id     = "temporary-jobs"
    status = "Enabled"
    filter { prefix = "users/" }
    expiration { days = 7 }
    noncurrent_version_expiration { noncurrent_days = 7 }
  }
}
resource "aws_ecr_repository" "app" {
  for_each             = toset(["api", "worker", "training"])
  name                 = "${local.name}/${each.key}"
  image_tag_mutability = "IMMUTABLE"
}
output "buckets" { value = { for k, v in terraform_data.buckets : k => v.output } }
output "repositories" { value = { for k, v in aws_ecr_repository.app : k => v.repository_url } }
