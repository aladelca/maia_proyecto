data "aws_caller_identity" "current" {}
locals {
  instance_profile = var.instance_profile != "" ? var.instance_profile : aws_iam_instance_profile.ec2[0].arn
  job_role         = var.job_role != "" ? var.job_role : aws_iam_role.job[0].arn
  tracking_role    = var.job_role != "" ? var.job_role : try(aws_iam_role.tracking[0].arn, "")
  research_arn     = "arn:aws:s3:::${var.inputs.buckets.research}"
  train_permissions = [
    { Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = local.research_arn },
    { Effect = "Allow", Action = ["s3:GetObjectVersion"], Resource = ["${local.research_arn}/datasets/*", "arn:aws:s3:::${var.inputs.buckets.releases}/releases/*"] },
    { Effect = "Allow", Action = ["s3:PutObject", "s3:GetObjectVersion", "s3:GetObject", "s3:AbortMultipartUpload"], Resource = ["${local.research_arn}/candidates/*", "${local.research_arn}/tracking/*"] },
    { Effect = "Allow", Action = ["sagemaker:CallMlflowAppApi", "sagemaker:DescribeMlflowApp"], Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:mlflow-app/*" }
  ]
}
resource "aws_iam_role" "job" {
  count              = var.job_role == "" ? 1 : 0
  name               = "${local.name}-job"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ecs-tasks.amazonaws.com" } }] })
}
resource "aws_iam_role_policy" "job" {
  count  = var.job_role == "" ? 1 : 0
  role   = aws_iam_role.job[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = local.train_permissions })
}
resource "aws_iam_role" "ec2" {
  count              = var.instance_profile == "" ? 1 : 0
  name               = "${local.name}-ec2"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ec2.amazonaws.com" } }] })
}
resource "aws_iam_instance_profile" "ec2" {
  count = var.instance_profile == "" ? 1 : 0
  name  = "${local.name}-ec2"
  role  = aws_iam_role.ec2[0].name
}
resource "aws_iam_role_policy_attachment" "ecs" {
  count      = var.instance_profile == "" ? 1 : 0
  role       = aws_iam_role.ec2[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}
resource "aws_iam_role_policy" "standalone" {
  count  = var.instance_profile == "" ? 1 : 0
  role   = aws_iam_role.ec2[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = local.train_permissions })
}
resource "aws_iam_role" "tracking" {
  count              = var.managed_mlflow && var.job_role == "" ? 1 : 0
  name               = "${local.name}-mlflow"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "sagemaker.amazonaws.com" } }] })
}
resource "aws_iam_role_policy" "tracking" {
  count = length(aws_iam_role.tracking)
  role  = aws_iam_role.tracking[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = local.research_arn },
    { Effect = "Allow", Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:AbortMultipartUpload"], Resource = "${local.research_arn}/tracking/managed/*" }
  ] })
}
