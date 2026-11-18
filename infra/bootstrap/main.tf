terraform {
  required_version = ">= 1.10, < 2.0"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
provider "aws" { region = var.region }
data "aws_caller_identity" "current" {}
locals { name = "maia-sleep-dev-${data.aws_caller_identity.current.account_id}-${var.region}" }
resource "terraform_data" "state" {
  input = "${local.name}-state"
  provisioner "local-exec" {
    command = "python3 ../scripts/ensure_bucket.py ${self.input} ${var.region}"
  }
  lifecycle { prevent_destroy = true }
}
resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = terraform_data.state.output
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_versioning" "state" {
  bucket = terraform_data.state.output
  versioning_configuration { status = "Enabled" }
}
output "bucket" { value = terraform_data.state.output }
