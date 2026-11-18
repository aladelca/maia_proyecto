terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
variable "cidr" { default = "10.88.0.0/24" }
provider "aws" { region = var.region }
data "aws_availability_zones" "available" { state = "available" }
resource "aws_vpc" "training" {
  cidr_block           = var.cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "maia-sleep-dev-training" }
}
resource "aws_internet_gateway" "egress" { vpc_id = aws_vpc.training.id }
resource "aws_subnet" "compute" {
  count                   = 2
  vpc_id                  = aws_vpc.training.id
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  cidr_block              = cidrsubnet(var.cidr, 2, count.index)
  map_public_ip_on_launch = true
}
resource "aws_route_table" "compute" {
  vpc_id = aws_vpc.training.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.egress.id
  }
}
resource "aws_route_table_association" "compute" {
  count          = length(aws_subnet.compute)
  subnet_id      = aws_subnet.compute[count.index].id
  route_table_id = aws_route_table.compute.id
}
output "vpc_id" { value = aws_vpc.training.id }
output "subnets" { value = aws_subnet.compute[*].id }
resource "aws_security_group" "standalone" {
  name        = "maia-sleep-dev-standalone"
  description = "Outbound only for ephemeral EC2 training"
  vpc_id      = aws_vpc.training.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
output "security_group" { value = aws_security_group.standalone.id }
