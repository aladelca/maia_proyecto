terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
variable "subnets" { type = list(string) }
variable "vpc_id" { type = string }
variable "instance_profile" { default = "" }
variable "job_role" { default = "" }
variable "batch_service_role" { default = null }
variable "image" { default = null }
variable "inputs" { type = any }
variable "managed_mlflow" { default = false }
variable "standalone_security_group" { type = string }
variable "use_batch" { default = true }
provider "aws" { region = var.region }
locals {
  name        = "maia-sleep-dev-training"
  environment = concat([{ name = "CLOUD_INPUTS", value = jsonencode(var.inputs) }, { name = "TRAIN_IMAGE", value = coalesce(var.image, "pending") }], var.managed_mlflow ? [{ name = "MLFLOW_TRACKING_URI", value = aws_sagemaker_mlflow_app.tracking[0].arn }] : [])
}
resource "aws_security_group" "compute" {
  count       = var.use_batch ? 1 : 0
  name        = local.name
  description = "Outbound access for EC2 training; no inbound connections"
  vpc_id      = var.vpc_id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_sagemaker_mlflow_app" "tracking" {
  count                   = var.managed_mlflow ? 1 : 0
  name                    = "maia-sleep-dev"
  artifact_store_uri      = "s3://${var.inputs.buckets.research}/tracking/managed"
  role_arn                = local.tracking_role
  model_registration_mode = "AutoModelRegistrationDisabled"
}
resource "aws_batch_compute_environment" "compute" {
  count        = var.use_batch ? 1 : 0
  name         = local.name
  type         = "MANAGED"
  service_role = var.batch_service_role
  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = 8
    instance_type       = ["m5.xlarge", "m6i.xlarge"]
    instance_role       = local.instance_profile
    security_group_ids  = [aws_security_group.compute[0].id]
    subnets             = var.subnets
  }
}
resource "aws_batch_job_queue" "training" {
  count    = var.use_batch ? 1 : 0
  name     = local.name
  state    = "ENABLED"
  priority = 1
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.compute[0].arn
  }
}
resource "aws_cloudwatch_log_group" "training" {
  name              = "/maia/sleep/dev/training"
  retention_in_days = 7
}
resource "aws_batch_job_definition" "fit" {
  count                 = var.use_batch && var.image != null ? 1 : 0
  name                  = local.name
  type                  = "container"
  platform_capabilities = ["EC2"]
  container_properties = jsonencode({
    image                = var.image, jobRoleArn = local.job_role, environment = local.environment,
    resourceRequirements = [{ type = "VCPU", value = "4" }, { type = "MEMORY", value = "12500" }],
    logConfiguration     = { logDriver = "awslogs", options = { "awslogs-group" = aws_cloudwatch_log_group.training.name, "awslogs-region" = var.region, "awslogs-stream-prefix" = "fit" } }
  })
  retry_strategy { attempts = 2 }
  timeout { attempt_duration_seconds = 43200 }
}
output "queue" { value = try(aws_batch_job_queue.training[0].arn, null) }
output "definition" { value = try(aws_batch_job_definition.fit[0].arn, null) }

output "runner" {
  value = { inputs = var.inputs, image = var.image, subnet = var.subnets[0], security_group = var.standalone_security_group, instance_profile = local.instance_profile,
  tracking = var.managed_mlflow ? aws_sagemaker_mlflow_app.tracking[0].arn : "", log_group = aws_cloudwatch_log_group.training.name }
}
