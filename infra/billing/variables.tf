variable "correo_de_avisos" {
  description = "Direccion a la que llegan las alertas. AWS manda un correo de confirmacion que HAY QUE aceptar: hasta entonces la alerta no sirve de nada."
  type        = string

  validation {
    condition     = can(regex("^[^@\s]+@[^@\s]+\.[^@\s]+$", var.correo_de_avisos))
    error_message = "Tiene que ser una direccion de correo valida."
  }
}

variable "limite_mensual" {
  description = "Techo mensual del presupuesto, en la moneda de la cuenta."
  type        = number
  default     = 5

  validation {
    condition     = var.limite_mensual > 0 && var.limite_mensual <= 50
    error_message = "Entre 1 y 50. Si necesitas mas, es que algo se ha ido de las manos y conviene mirarlo antes de subir el techo."
  }
}

variable "moneda" {
  description = "Moneda de facturacion de la cuenta. AWS factura en USD salvo que hayas cambiado la preferencia; si pones una que no coincide, el presupuesto no cuadra."
  type        = string
  default     = "USD"

  validation {
    condition     = contains(["USD", "EUR"], var.moneda)
    error_message = "USD o EUR."
  }
}
