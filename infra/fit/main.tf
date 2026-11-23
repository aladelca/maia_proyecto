terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
provider "aws" { region = var.region }
variable "runner" { type = any }
variable "attempt" {
  type = string
  validation {
    condition     = can(regex("^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$", var.attempt))
    error_message = "The attempt must have a UUID."
  }
}
variable "run_id" {
  type = string

  validation {
    condition     = can(regex("^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$", var.run_id))
    error_message = "Use a unique lower-case UUID per training campaign."
  }
}
variable "mode" {
  default = "stack"
  validation {
    condition     = contains(["stack", "refit", "tune"], var.mode)
    error_message = "Choose stack, refit or tune."
  }
}
variable "instance_type" { default = "t3.large" }
data "aws_ssm_parameter" "ecs_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2023/recommended/image_id"
}
resource "aws_instance" "fit" {

  ami                                  = data.aws_ssm_parameter.ecs_ami.value
  instance_type                        = var.instance_type
  subnet_id                            = var.runner.subnet
  associate_public_ip_address          = true
  vpc_security_group_ids               = [var.runner.security_group]
  iam_instance_profile                 = element(reverse(split("/", var.runner.instance_profile)), 0)
  instance_initiated_shutdown_behavior = "terminate"
  user_data_replace_on_change          = true
  user_data = templatefile("${path.module}/training.sh.tftpl", {
    region   = var.region, image = var.runner.image, registry = split("/", var.runner.image)[0], inputs = base64encode(jsonencode(var.runner.inputs)), attempt = var.attempt, job = var.run_id, mode = var.mode,
    bucket   = var.runner.inputs.buckets.research, group = var.runner.log_group,
    tracking = var.runner.tracking
  })
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 40
    encrypted             = true
    delete_on_termination = true
  }
  tags = { Name = "maia-sleep-dev-ephemeral-fit", Campaign = var.run_id }
}
output "instance_id" { value = aws_instance.fit.id }
