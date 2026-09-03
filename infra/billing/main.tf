// Alertas de facturacion.
//
// Este modulo se aplica ANTES que ningun otro y vive en su propio estado, no
// junto al resto de la infraestructura. El motivo es concreto: si estuviera en
// el mismo estado, un `terraform destroy` para dejar de pagar se llevaria por
// delante las alertas que avisan de que estas pagando. La red de seguridad no
// puede compartir ciclo de vida con lo que vigila.
//
// No crea nada que cueste dinero. Los dos primeros presupuestos de una cuenta
// son gratis, y este modulo crea exactamente dos.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  // AWS Budgets es un servicio global cuyo endpoint vive en us-east-1. Que
  // este modulo apunte aqui no tiene nada que ver con donde va a correr la
  // aplicacion: eso se decide en el modulo de la aplicacion.
  region = "us-east-1"

  default_tags {
    tags = {
      Project   = "imagent"
      ManagedBy = "terraform"
    }
  }
}

// ---------------------------------------------------------------------------
// Aviso al primer euro
// ---------------------------------------------------------------------------
resource "aws_budgets_budget" "primer_gasto" {
  name         = "imagent-primer-gasto"
  budget_type  = "COST"
  limit_amount = "1"
  limit_unit   = var.moneda
  time_unit    = "MONTHLY"

  // La pregunta que responde este presupuesto no es "¿me estoy pasando?" sino
  // "¿ha empezado a costar algo?". En una cuenta que deberia estar casi a cero,
  // el primer euro ya es informacion: significa que algo esta encendido que no
  // creias tener encendido.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.correo_de_avisos]
  }
}

// ---------------------------------------------------------------------------
// Presupuesto mensual
// ---------------------------------------------------------------------------
resource "aws_budgets_budget" "mensual" {
  name         = "imagent-mensual"
  budget_type  = "COST"
  limit_amount = tostring(var.limite_mensual)
  limit_unit   = var.moneda
  time_unit    = "MONTHLY"

  // Avisos escalonados sobre el gasto REAL.
  dynamic "notification" {
    for_each = [50, 80, 100]

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.correo_de_avisos]
    }
  }

  // Y uno sobre la PREVISION, que es el unico que avisa a tiempo.
  //
  // Un aviso sobre gasto real al 100% llega cuando ya te lo has gastado. Este
  // salta cuando la proyeccion del mes se pasa del limite, o sea el dia 3 si
  // te has dejado un NAT Gateway encendido, no el dia 28.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.correo_de_avisos]
  }
}
