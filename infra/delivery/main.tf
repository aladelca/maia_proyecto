terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
variable "repository" { type = string }
variable "research_bucket" { type = string }
variable "connection_arn" { default = "" }
variable "build_role_arn" { default = "" }
variable "release_role_arn" { default = "" }
variable "enabled" { default = false }
provider "aws" { region = var.region }
resource "aws_codestarconnections_connection" "github" {
  count         = var.connection_arn == "" ? 1 : 0
  name          = "maia-sleep-dev"
  provider_type = "GitHub"
}
locals {
  connection = var.connection_arn != "" ? var.connection_arn : aws_codestarconnections_connection.github[0].arn
  stages     = { Build = "build", Release = "release" }
  name       = "maia-sleep-dev-delivery"
}
# Preserve any existing secret when handing web deployments to Vercel.
removed {
  from = aws_secretsmanager_secret.vercel
  lifecycle { destroy = false }
}
resource "aws_codebuild_project" "stage" {
  for_each      = var.enabled ? local.stages : {}
  name          = "${local.name}-${each.value}"
  service_role  = local.codebuild_roles[each.key]
  build_timeout = 45
  artifacts { type = "CODEPIPELINE" }
  source {
    type      = "CODEPIPELINE"
    buildspec = "ci/${each.value}.yml"
  }
  environment {
    type            = "LINUX_CONTAINER"
    image           = "aws/codebuild/standard:7.0"
    compute_type    = "BUILD_GENERAL1_MEDIUM"
    privileged_mode = each.key == "Build"
    environment_variable {
      name  = "RESEARCH_BUCKET"
      value = var.research_bucket
    }
  }
}
resource "aws_iam_role" "pipeline" {
  count              = var.enabled ? 1 : 0
  name               = local.name
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "codepipeline.amazonaws.com" } }] })
}
resource "aws_iam_role_policy" "pipeline" {
  count = var.enabled ? 1 : 0
  role  = aws_iam_role.pipeline[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["codeconnections:UseConnection", "codestar-connections:UseConnection"], Resource = local.connection },
    { Effect = "Allow", Action = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"], Resource = [for s in aws_codebuild_project.stage : s.arn] },
    { Effect = "Allow", Action = ["s3:GetBucketVersioning", "s3:GetBucketLocation"], Resource = "arn:aws:s3:::${var.research_bucket}" },
    { Effect = "Allow", Action = ["s3:PutObject", "s3:GetObject", "s3:GetObjectVersion"], Resource = "arn:aws:s3:::${var.research_bucket}/${local.name}/*" }
  ] })
}
resource "aws_codepipeline" "dev" {
  count          = var.enabled ? 1 : 0
  name           = local.name
  role_arn       = aws_iam_role.pipeline[0].arn
  pipeline_type  = "V2"
  execution_mode = "QUEUED"
  artifact_store {
    location = var.research_bucket
    type     = "S3"
  }
  stage {
    name = "Source"
    action {
      name             = "GitHub"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      output_artifacts = ["source"]
      configuration    = { ConnectionArn = local.connection, FullRepositoryId = var.repository, BranchName = "main" }
    }
  }
  stage {
    name = "Build"
    action {
      name             = "ValidateAndBuild"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      input_artifacts  = ["source"]
      output_artifacts = ["images"]
      configuration    = { ProjectName = aws_codebuild_project.stage["Build"].name }
    }
  }
  stage {
    name = "Release"
    action {
      name            = "Deploy"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      input_artifacts = ["source", "images"]
      configuration   = { ProjectName = aws_codebuild_project.stage["Release"].name, PrimarySource = "source" }
    }
  }
}
output "connection" { value = local.connection }
