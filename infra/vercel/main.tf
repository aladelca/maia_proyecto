terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { vercel = { source = "vercel/vercel", version = "~> 4.0" } }
}
variable "team_id" { type = string }
variable "project_name" { type = string }
variable "publish" { default = true }
provider "vercel" { team = var.team_id }
resource "vercel_project" "web" {
  name    = var.project_name
  team_id = var.team_id
}
data "vercel_project_directory" "site" {
  count = var.publish ? 1 : 0
  path  = "../../frontend/dist"
}
resource "vercel_deployment" "live" {
  count       = var.publish ? 1 : 0
  project_id  = vercel_project.web.id
  team_id     = var.team_id
  files       = data.vercel_project_directory.site[0].files
  path_prefix = data.vercel_project_directory.site[0].path
  production  = true
  lifecycle { create_before_destroy = true }
}
output "project" { value = vercel_project.web.name }
output "deployment" { value = try(vercel_deployment.live[0].url, null) }
