output "siguiente_paso" {
  description = "Lo que hay que hacer a mano despues de aplicar esto."
  value       = <<-TEXTO

    Presupuestos creados. Quedan DOS cosas que no se pueden automatizar:

    1. Revisa tu correo (${var.correo_de_avisos}) y confirma la suscripcion.
       Hasta que no lo hagas, las alertas no llegan a ninguna parte.

    2. En la consola, Billing -> Preferencias -> activa "Free tier usage
       alerts". Son avisos distintos: estos saltan cuando te acercas al limite
       de la capa gratuita de un servicio concreto, no cuando gastas dinero.

    Ten en cuenta que los datos de facturacion de AWS llegan con hasta 24h de
    retraso. Esto es una red de seguridad, no un cortacircuitos: no impide que
    se gaste, avisa despues. Por eso el resto de la infraestructura se diseña
    para no poder gastar mucho, en vez de confiar en que esto avise a tiempo.
  TEXTO
}
