variable "monthly_budget" {
  description = "Optional AWS account budget in USD. Supply an approved cap and notification recipient to activate it."
  type        = object({ limit_usd = number, recipient = string })
  default     = null
  validation {
    condition     = var.monthly_budget == null ? true : var.monthly_budget.limit_usd > 0 && can(regex("^[^ @]+@[^ @]+[.][^ @]+$", var.monthly_budget.recipient))
    error_message = "A budget requires a positive limit and a valid recipient."
  }
}
resource "aws_budgets_budget" "account" {
  count        = var.monthly_budget == null ? 0 : 1
  name         = "maia-sleep-dev-account-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget.limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    notification_type          = "ACTUAL"
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    subscriber_email_addresses = [var.monthly_budget.recipient]
  }
}
