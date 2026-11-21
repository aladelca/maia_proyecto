data "aws_caller_identity" "current" {}
locals {
  account         = data.aws_caller_identity.current.account_id
  state           = "maia-sleep-dev-${local.account}-${var.region}-state"
  ecr             = "arn:aws:ecr:${var.region}:${local.account}:repository/maia-sleep-dev/*"
  create_roles    = toset([for k, v in { Build = var.build_role_arn, Release = var.release_role_arn } : k if var.enabled && v == ""])
  codebuild_roles = { for k, v in { Build = var.build_role_arn, Release = var.release_role_arn } : k => v != "" ? v : try(aws_iam_role.codebuild[k].arn, "") }
  common = [
    { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:${var.region}:${local.account}:log-group:/aws/codebuild/${local.name}-*" },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"], Resource = "arn:aws:s3:::${var.research_bucket}/${local.name}/*" },
    { Effect = "Allow", Action = ["s3:GetBucketLocation", "s3:GetBucketVersioning", "s3:ListBucket"], Resource = ["arn:aws:s3:::${local.state}", "arn:aws:s3:::${var.research_bucket}"] }
  ]
  build = [
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "arn:aws:s3:::${local.state}/platform/terraform.tfstate" },
    { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
    { Effect = "Allow", Action = ["ecr:DescribeImages", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage"], Resource = local.ecr }
  ]
}
variable "deployment_role_arns" {
  type        = list(string)
  default     = []
  description = "Existing execution roles assigned to the managed Lambda, Step Functions and training jobs."
}
resource "aws_iam_role" "codebuild" {
  for_each           = local.create_roles
  name               = "${local.name}-${lower(each.key)}"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "codebuild.amazonaws.com" } }] })
}
resource "aws_iam_role_policy" "codebuild" {
  for_each = local.create_roles
  role     = aws_iam_role.codebuild[each.key].id
  policy = jsonencode({ Version = "2012-10-17", Statement = concat(local.common, jsondecode(each.key == "Build" ? jsonencode(local.build) : jsonencode(concat([
    { Effect = "Allow", Action = ["ecr:GetRepositoryPolicy", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], Resource = local.ecr },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion"], Resource = "arn:aws:s3:::${var.research_bucket}/delivery/config/*" },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = [for k in ["aws", "training"] : "arn:aws:s3:::${local.state}/${k}/terraform.tfstate"] },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = [for k in ["aws", "training"] : "arn:aws:s3:::${local.state}/${k}/terraform.tfstate.tflock"] },
    { Effect = "Allow", Action = ["lambda:UpdateFunctionCode", "lambda:UpdateFunctionConfiguration", "lambda:PublishVersion", "lambda:Get*", "lambda:List*"], Resource = "arn:aws:lambda:${var.region}:${local.account}:function:maia-sleep-dev-*" },
    { Effect = "Allow", Action = ["states:UpdateStateMachine", "states:PublishStateMachineVersion", "states:UpdateStateMachineAlias", "states:Describe*", "states:List*"], Resource = "arn:aws:states:${var.region}:${local.account}:stateMachine:maia-sleep-dev*" },
    { Effect = "Allow", Action = ["batch:RegisterJobDefinition", "batch:DeregisterJobDefinition", "batch:TagResource"], Resource = ["arn:aws:batch:${var.region}:${local.account}:job-definition/maia-sleep-dev-training*"] },
    { Effect = "Allow", Action = ["batch:Describe*", "batch:ListTagsForResource", "s3:GetBucketCORS", "cloudwatch:DescribeAlarms", "cloudwatch:ListTagsForResource", "apigateway:GET", "cognito-idp:Describe*", "cognito-idp:ListTagsForResource", "ec2:Describe*", "sagemaker:DescribeMlflowApp", "sagemaker:ListTags", "logs:DescribeLogGroups", "logs:ListTags*", "iam:GetRole", "iam:GetRolePolicy", "iam:ListAttachedRolePolicies", "iam:ListRolePolicies", "iam:GetInstanceProfile"], Resource = "*" }
  ], length(var.deployment_role_arns) == 0 ? [] : [{ Effect = "Allow", Action = "iam:PassRole", Resource = var.deployment_role_arns, Condition = { StringEquals = { "iam:PassedToService" = ["lambda.amazonaws.com", "states.amazonaws.com", "ecs-tasks.amazonaws.com"] } } }])))) })
}
