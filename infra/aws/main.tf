terraform {
  required_version = ">= 1.10, < 2"
  backend "s3" {}
  required_providers { aws = { source = "hashicorp/aws", version = "~> 6.0" } }
}
variable "region" { default = "us-west-2" }
variable "live_version_arn" { default = null }
variable "origin" { type = string }
variable "images" { type = map(string) }
variable "inputs" { type = any }
variable "existing_role_arn" { default = "" }
provider "aws" { region = var.region }
data "aws_caller_identity" "current" {}
locals {
  name        = "maia-sleep-dev"
  bucket      = var.inputs.buckets.jobs
  release     = var.inputs.buckets.releases
  obj         = var.inputs.objects
  machine_arn = "arn:aws:states:${var.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${local.name}"
  components  = toset(["api", "worker", "states"])
  roles       = { for c in local.components : c => (var.existing_role_arn != "" ? var.existing_role_arn : aws_iam_role.component[c].arn) }
}
resource "aws_iam_role" "component" {
  for_each           = var.existing_role_arn == "" ? local.components : toset([])
  name               = "${local.name}-${each.key}"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = each.key == "states" ? "states.amazonaws.com" : "lambda.amazonaws.com" } }] })
}
locals {
  api_policy = [
    { Effect = "Allow", Action = ["s3:GetObjectVersion", "s3:PutObject"], Resource = "arn:aws:s3:::${local.bucket}/users/*/uploads/*" },
    { Effect = "Allow", Action = ["s3:GetObject"], Resource = "arn:aws:s3:::${local.bucket}/users/*/results/*" },
    { Effect = "Allow", Action = ["states:StartExecution"], Resource = "${local.machine_arn}:live" },
    { Effect = "Allow", Action = ["states:DescribeExecution"], Resource = "${replace(local.machine_arn, ":stateMachine:", ":execution:")}:*" }
  ]
  worker_policy = [
    { Effect = "Allow", Action = ["s3:GetObjectVersion"], Resource = ["arn:aws:s3:::${local.release}/releases/*", "arn:aws:s3:::${local.bucket}/users/*/uploads/*"] },
    { Effect = "Allow", Action = ["s3:PutObject"], Resource = "arn:aws:s3:::${local.bucket}/users/*/results/*" }
  ]
}
resource "aws_iam_role_policy" "api" {
  count  = var.existing_role_arn == "" ? 1 : 0
  role   = aws_iam_role.component["api"].id
  policy = jsonencode({ Version = "2012-10-17", Statement = local.api_policy })
}
resource "aws_iam_role_policy" "worker" {
  count  = var.existing_role_arn == "" ? 1 : 0
  role   = aws_iam_role.component["worker"].id
  policy = jsonencode({ Version = "2012-10-17", Statement = local.worker_policy })
}
resource "aws_iam_role_policy" "states" {
  count  = var.existing_role_arn == "" ? 1 : 0
  role   = aws_iam_role.component["states"].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = "lambda:InvokeFunction", Resource = "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.name}-worker:*" }] })
}
resource "aws_iam_role_policy_attachment" "logging" {
  for_each   = var.existing_role_arn == "" ? toset(["api", "worker"]) : toset([])
  role       = aws_iam_role.component[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
resource "aws_cloudwatch_log_group" "app" {
  for_each          = toset(["api", "worker"])
  name              = "/aws/lambda/${local.name}-${each.key}"
  retention_in_days = 7
}
resource "aws_lambda_function" "app" {
  for_each      = toset(["api", "worker"])
  function_name = "${local.name}-${each.key}"
  role          = local.roles[each.key]
  package_type  = "Image"
  image_uri     = var.images[each.key]
  architectures = ["x86_64"]
  publish       = true
  memory_size   = each.key == "worker" ? 3072 : 512
  timeout       = each.key == "worker" ? 120 : 20
  environment {
    variables = merge({ JOBS_BUCKET = local.bucket }, each.key == "api" ? {
      MACHINE_ARN = local.machine_arn, EXAMPLE_KEY = local.obj["example.parquet"].key, EXAMPLE_VERSION = local.obj["example.parquet"].version, EXAMPLE_SHA256 = local.obj["example.parquet"].sha256
      } : {
      RELEASE_BUCKET = local.release, MANIFEST_KEY = local.obj.manifest.key, MANIFEST_VERSION = local.obj.manifest.version, MANIFEST_SHA256 = local.obj.manifest.sha256, OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "1"
    })
  }
  depends_on = [aws_cloudwatch_log_group.app, aws_iam_role_policy_attachment.logging]
}
resource "aws_sfn_state_machine" "predict" {
  name     = local.name
  role_arn = local.roles["states"]
  publish  = true
  type     = "STANDARD"
  definition = jsonencode({ StartAt = "Predict", TimeoutSeconds = 420, States = { Predict = {
    Type  = "Task", Resource = aws_lambda_function.app["worker"].qualified_arn, End = true,
    Retry = [{ ErrorEquals = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException", "Lambda.TooManyRequestsException"], IntervalSeconds = 2, MaxAttempts = 3, BackoffRate = 2 }]
  } } })
}
resource "aws_sfn_alias" "live" {
  name = "live"
  routing_configuration {
    state_machine_version_arn = coalesce(var.live_version_arn, aws_sfn_state_machine.predict.state_machine_version_arn)
    weight                    = 100
  }
}
resource "aws_cognito_user_pool" "users" {
  name = local.name
  admin_create_user_config { allow_admin_create_user_only = true }
  username_configuration { case_sensitive = false }
}
resource "aws_cognito_user_pool_client" "web" {
  name                                 = "sleep-web"
  user_pool_id                         = aws_cognito_user_pool.users.id
  generate_secret                      = false
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  supported_identity_providers         = ["COGNITO"]
  allowed_oauth_scopes                 = ["openid", "aws.cognito.signin.user.admin"]
  callback_urls                        = ["${var.origin}/"]
  logout_urls                          = ["${var.origin}/"]
  explicit_auth_flows                  = ["ALLOW_USER_SRP_AUTH", "ALLOW_REFRESH_TOKEN_AUTH", "ALLOW_USER_PASSWORD_AUTH"]
  prevent_user_existence_errors        = "ENABLED"
}
resource "aws_cognito_user_pool_domain" "login" {
  domain       = "${local.name}-${data.aws_caller_identity.current.account_id}"
  user_pool_id = aws_cognito_user_pool.users.id
}
resource "aws_apigatewayv2_api" "api" {
  name          = local.name
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = [var.origin]
    allow_headers = ["authorization", "content-type", "idempotency-key"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    max_age       = 300
  }
}
resource "aws_apigatewayv2_authorizer" "users" {
  api_id           = aws_apigatewayv2_api.api.id
  authorizer_type  = "JWT"
  name             = "sleep-users"
  identity_sources = ["$request.header.Authorization"]
  jwt_configuration {
    audience = [aws_cognito_user_pool_client.web.id]
    issuer   = "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.users.id}"
  }
}
resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.app["api"].invoke_arn
  payload_format_version = "2.0"
}
resource "aws_apigatewayv2_route" "routes" {
  for_each             = toset(["GET /v1/examples/default", "POST /v1/uploads", "POST /v1/predictions", "GET /v1/predictions/{job}"])
  api_id               = aws_apigatewayv2_api.api.id
  route_key            = each.key
  target               = "integrations/${aws_apigatewayv2_integration.api.id}"
  authorization_type   = "JWT"
  authorizer_id        = aws_apigatewayv2_authorizer.users.id
  authorization_scopes = ["aws.cognito.signin.user.admin"]
}
resource "aws_apigatewayv2_stage" "live" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 15
    throttling_rate_limit  = 10
  }
}
resource "aws_lambda_permission" "invoke" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app["api"].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
resource "aws_s3_bucket_cors_configuration" "jobs" {
  bucket = local.bucket
  cors_rule {
    allowed_origins = [var.origin]
    allowed_methods = ["GET", "POST", "HEAD"]
    allowed_headers = ["*"]
    expose_headers  = ["x-amz-version-id", "ETag"]
    max_age_seconds = 300
  }
}
output "public_config" {
  value = { api = aws_apigatewayv2_api.api.api_endpoint, issuer = "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.users.id}", clientId = aws_cognito_user_pool_client.web.id, login = "https://${aws_cognito_user_pool_domain.login.domain}.auth.${var.region}.amazoncognito.com", origin = var.origin }
}
output "user_pool" { value = aws_cognito_user_pool.users.id }

output "workflow_version" { value = aws_sfn_state_machine.predict.state_machine_version_arn }
