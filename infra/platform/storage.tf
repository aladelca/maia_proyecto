resource "aws_s3_bucket_server_side_encryption_configuration" "private" {
  for_each = terraform_data.buckets
  bucket   = each.value.output
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_policy" "tls" {
  for_each = terraform_data.buckets
  bucket   = each.value.output
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Sid       = "RequireTLS", Effect = "Deny", Principal = "*", Action = "s3:*",
    Resource  = ["arn:aws:s3:::${each.value.output}", "arn:aws:s3:::${each.value.output}/*"],
    Condition = { Bool = { "aws:SecureTransport" = "false" } }
  }] })
}
