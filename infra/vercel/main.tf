terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { vercel = { source = "vercel/vercel", version = "~> 4.0" } }
}
variable "team_id" { type = string }
variable "project_name" { type = string }
# Older saved tfvars contain publish. Git now controls publication.
variable "publish" { default = true }
variable "repository" { default = "aladelca/maia_proyecto" }
provider "vercel" { team = var.team_id }

resource "vercel_project" "web" {
  name                                 = var.project_name
  team_id                              = var.team_id
  root_directory                       = "frontend"
  framework                            = "vite"
  install_command                      = "npm ci"
  output_directory                     = "dist"
  node_version                         = "22.x"
  enable_affected_projects_deployments = false
  git_repository = {
    type              = "github"
    repo              = var.repository
    production_branch = "main"
  }
  # Operators set the browser configuration in Vercel for Production and Preview.
  lifecycle { ignore_changes = [environment] }
}

# Git takes over the production alias. Retain the previous release for rollback.
removed {
  from = vercel_deployment.live
  lifecycle { destroy = false }
}

output "project" { value = vercel_project.web.name }
output "deployment" { value = "${vercel_project.web.name}.vercel.app" }
