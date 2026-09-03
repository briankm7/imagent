# Infraestructura

Terraform. Tres estados separados, aplicados en este orden:

| Módulo | Qué hace | Coste |
|---|---|---|
| `billing/` | presupuestos y alertas de gasto | 0 € |
| `bootstrap/` | backend de estado y rol OIDC para GitHub Actions | céntimos |
| `app/` | ECR, red, ECS Fargate, S3, DynamoDB, secretos | 0 € apagado |

**Por qué tres estados y no uno.** `billing/` vigila lo que gasta `app/`. Si
compartieran estado, un `terraform destroy` para dejar de pagar se llevaría por
delante las alertas que avisan de que estás pagando. La red de seguridad no
puede tener el mismo ciclo de vida que aquello que vigila.

---

## Lo que hay que hacer a mano (una vez)

No se puede aplicar Terraform en una cuenta que todavía no existe. Estas tres
cosas son el mínimo irreducible:

**1. Crear la cuenta de AWS.** <https://portal.aws.amazon.com/billing/signup>.
Pide tarjeta aunque no se cobre nada.

**2. Activar MFA en el usuario root y no volver a usarlo.** Consola → arriba a
la derecha, tu nombre → *Security credentials* → *Assign MFA device*. El usuario
root puede borrar la cuenta entera y no tiene límites; a partir de aquí no se
toca.

**3. Crear un usuario IAM para ti, con MFA y claves de acceso.**
IAM → *Users* → *Create user* → adjunta `AdministratorAccess` → luego, dentro
del usuario, *Security credentials* → *Create access key* → tipo *CLI*.

Guarda las dos claves. Configúralas con `aws configure`.

**4. Dar acceso de IAM a los datos de facturación.** Consola → *Billing* →
*Preferences* → *IAM user and role access to Billing information* → activar.
Sin esto, el usuario IAM no puede crear presupuestos aunque sea administrador.

---

## Bloque 1: las alertas de facturación

Antes de crear nada más.

```bash
cd infra/billing
cp terraform.tfvars.example terraform.tfvars   # y pon tu correo
terraform init
terraform plan       # revisa: deben salir 2 recursos, ninguno de pago
terraform apply
```

Después:

- **Confirma la suscripción en el correo que te llega.** Hasta que no lo hagas,
  las alertas no van a ninguna parte. Este es el paso que todo el mundo se salta.
- Activa los *Free tier usage alerts* en Billing → Preferences.

### Qué crea, y qué no

Dos presupuestos:

- **`imagent-primer-gasto`** (1 €). No responde a "¿me estoy pasando?" sino a
  "¿ha empezado a costar algo?". En una cuenta que debería estar casi a cero, el
  primer euro ya es información: algo está encendido que no creías tener
  encendido.
- **`imagent-mensual`** (5 € por defecto). Avisos al 50 %, 80 % y 100 % del
  gasto real, **más uno sobre la previsión**. El de previsión es el único que
  avisa a tiempo: el de gasto real al 100 % llega cuando ya te lo has gastado; el
  de previsión salta el día 3 si te dejaste algo caro encendido, no el día 28.

Los dos primeros presupuestos de una cuenta son gratis. Este módulo crea
exactamente dos.

### Limitación que hay que tener clara

**Los datos de facturación de AWS llegan con hasta 24 horas de retraso.** Esto es
una red de seguridad, no un cortacircuitos: no impide gastar, avisa después. Por
eso el resto de la infraestructura está diseñada para *no poder* gastar mucho
—sin NAT Gateway, sin balanceador, con el servicio apagado por defecto— en vez
de confiar en que la alerta llegue a tiempo.

---

## Las tres trampas de coste que este diseño evita

| Recurso | Coste | Por qué no está |
|---|---|---|
| **NAT Gateway** | ~32 €/mes | el servicio va en subred pública con IP propia |
| **Application Load Balancer** | ~18 €/mes | se accede por IP; sin HTTPS, pero sin factura |
| **Tarea encendida sin darte cuenta** | ~8 €/mes | `desired_count = 0` por defecto |

Las tres juntas serían ~58 €/mes por un proyecto que nadie está usando.
