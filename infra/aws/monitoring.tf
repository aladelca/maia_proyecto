locals {
  alarms = {
    "worker-errors"   = { namespace = "AWS/Lambda", metric = "Errors", dimension = "FunctionName", value = aws_lambda_function.app["worker"].function_name }
    "api-throttles"   = { namespace = "AWS/Lambda", metric = "Throttles", dimension = "FunctionName", value = aws_lambda_function.app["api"].function_name }
    "workflow-failed" = { namespace = "AWS/States", metric = "ExecutionsFailed", dimension = "StateMachineArn", value = aws_sfn_state_machine.predict.arn }
  }
}
resource "aws_cloudwatch_metric_alarm" "service" {
  for_each            = local.alarms
  alarm_name          = "${local.name}-${each.key}"
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  dimensions          = { (each.value.dimension) = each.value.value }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
}
