# One private machine for the full run. See deploy/physionet/README.md.
#
# No inbound port is open: you reach it through AWS Systems Manager. The disk is encrypted, and
# the PhysioNet password lives in Parameter Store, set by you outside Terraform, so it never
# enters a state file. An idle alarm stops the instance, and the disk survives a stop.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  type        = string
  description = "The run is CPU-bound and peaks near 8 GB. More cores shorten the booster fits."
  default     = "m7i.4xlarge"
}

variable "disk_gb" {
  type    = number
  default = 120
}

variable "monthly_budget_usd" {
  type    = number
  default = 100
}

variable "alert_email" {
  type = string
}

variable "repo_url" {
  type    = string
  default = "https://github.com/Arnavnemade1/Sepsis.git"
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "aegis-physionet", DataClass = "physionet-credentialed" }
  }
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_caller_identity" "me" {}

# Outbound only: physionet.org, package mirrors, the SSM endpoints. Nothing can connect in.
resource "aws_security_group" "run" {
  name_prefix = "aegis-physionet-"
  description = "No inbound. Outbound HTTPS only."
  vpc_id      = data.aws_vpc.default.id
  egress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "HTTP for the OS package mirrors"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "run" {
  name_prefix        = "aegis-physionet-"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.run.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# The instance may read exactly the two parameters holding the PhysioNet login.
data "aws_iam_policy_document" "secrets" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter/physionet/*"]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "secrets" {
  role   = aws_iam_role.run.id
  policy = data.aws_iam_policy_document.secrets.json
}

resource "aws_iam_instance_profile" "run" {
  name_prefix = "aegis-physionet-"
  role        = aws_iam_role.run.name
}

resource "aws_instance" "run" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.instance_type
  iam_instance_profile        = aws_iam_instance_profile.run.name
  vpc_security_group_ids      = [aws_security_group.run.id]
  associate_public_ip_address = true
  monitoring                  = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.disk_gb
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", { repo_url = var.repo_url })

  lifecycle {
    ignore_changes = [ami, user_data]
  }
}

# Stops (never terminates) the machine after two idle hours. A stop keeps the lake.
resource "aws_cloudwatch_metric_alarm" "idle" {
  alarm_name          = "aegis-physionet-idle"
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  dimensions          = { InstanceId = aws_instance.run.id }
  statistic           = "Average"
  comparison_operator = "LessThanThreshold"
  threshold           = 3
  period              = 300
  evaluation_periods  = 24
  alarm_actions       = ["arn:aws:automate:${var.region}:ec2:stop"]
}

resource "aws_budgets_budget" "run" {
  name         = "aegis-physionet"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}

output "instance_id" {
  value = aws_instance.run.id
}

output "connect" {
  value = "aws ssm start-session --region ${var.region} --target ${aws_instance.run.id}"
}
