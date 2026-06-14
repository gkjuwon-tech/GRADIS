# -*- coding: utf-8 -*-
"""
Generate GRADIS pilot-deployment Letters of Intent for 3 target countries.
Structure faithfully replicates the public eForms "LETTER OF INTENT (TRANSACTION)"
template (sections I-VIII, Buyer/Seller, Goods/Services, Payment, Deposit,
Financing, Currency, Governing Law, signature blocks). Only the content is filled.
Buyer = local public-safety agency / security operator. Seller = GRADIS.
PH/ZA = English, MX = Spanish.
"""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

NAVY = HexColor("#0F2A43")
ACCENT = HexColor("#1F7A6E")
GREY = HexColor("#555555")

styles = getSampleStyleSheet()
S = {
    "brand": ParagraphStyle("brand", parent=styles["Normal"], fontName="Helvetica-Bold",
                            fontSize=16, textColor=NAVY, leading=18),
    "tag": ParagraphStyle("tag", parent=styles["Normal"], fontName="Helvetica-Oblique",
                          fontSize=8, textColor=GREY, leading=10),
    "title": ParagraphStyle("title", parent=styles["Normal"], fontName="Helvetica-Bold",
                            fontSize=13, textColor=NAVY, alignment=TA_CENTER, leading=16,
                            spaceBefore=4, spaceAfter=2),
    "sub": ParagraphStyle("sub", parent=styles["Normal"], fontName="Helvetica-Bold",
                          fontSize=10, textColor=ACCENT, alignment=TA_CENTER, leading=12),
    "meta": ParagraphStyle("meta", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=9.5, leading=14),
    "body": ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=5),
    "sec": ParagraphStyle("sec", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
                          leftIndent=14, firstLineIndent=-14),
    "small": ParagraphStyle("small", parent=styles["Normal"], fontName="Helvetica-Oblique",
                            fontSize=7.5, textColor=GREY, leading=9.5),
    "sign": ParagraphStyle("sign", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=9.5, leading=18),
}

CB_OFF = "[  ]"
CB_ON = "[ X ]"


def build(path, L):
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                            topMargin=0.7*inch, bottomMargin=0.6*inch,
                            leftMargin=0.85*inch, rightMargin=0.85*inch,
                            title=L["doc_title"], author="GRADIS")
    e = []
    # ---- Letterhead ----
    e.append(Paragraph("GRADIS", S["brand"]))
    e.append(Paragraph("Guardian Recon Aerial Drone for Incident Sensing &nbsp;|&nbsp; "
                       "Privacy-preserving aerial public-safety platform", S["tag"]))
    e.append(Spacer(1, 4))
    e.append(HRFlowable(width="100%", thickness=1.1, color=NAVY,
                        spaceBefore=2, spaceAfter=8))
    # ---- Title ----
    e.append(Paragraph(L["title"], S["title"]))
    e.append(Paragraph(L["subtitle"], S["sub"]))
    e.append(Spacer(1, 8))
    # ---- Meta ----
    e.append(Paragraph(L["eff"], S["meta"]))
    e.append(Paragraph(L["re"], S["meta"]))
    e.append(Spacer(1, 6))
    e.append(Paragraph(L["intro"], S["body"]))
    e.append(Spacer(1, 2))
    # ---- Sections ----
    for sec in L["sections"]:
        e.append(Paragraph(sec, S["sec"]))
    e.append(Spacer(1, 6))
    e.append(Paragraph(L["context_label"], S["meta"]))
    e.append(Paragraph(L["context"], S["body"]))
    e.append(Spacer(1, 10))
    # ---- Signature blocks ----
    e.append(HRFlowable(width="100%", thickness=0.6, color=GREY,
                        spaceBefore=2, spaceAfter=8))
    sig_tbl = Table([
        [Paragraph(L["buyer_hdr"], S["sec"]), Paragraph(L["seller_hdr"], S["sec"])],
        [Paragraph(L["sig_line"], S["sign"]), Paragraph(L["sig_line"], S["sign"])],
        [Paragraph(L["name_line"], S["sign"]), Paragraph(L["name_line"], S["sign"])],
        [Paragraph(L["title_line"], S["sign"]), Paragraph(L["title_line2"], S["sign"])],
        [Paragraph(L["date_line"], S["sign"]), Paragraph(L["date_line"], S["sign"])],
    ], colWidths=[3.1*inch, 3.1*inch])
    sig_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    e.append(sig_tbl)
    e.append(Spacer(1, 10))
    e.append(Paragraph(L["footer"], S["small"]))
    doc.build(e)
    print("wrote", path)


# ============================ PHILIPPINES (EN) ============================
PH = {
    "doc_title": "GRADIS Letter of Intent - Philippines",
    "title": "LETTER OF INTENT",
    "subtitle": "(PILOT DEPLOYMENT &mdash; TRANSACTION)",
    "eff": "<b>Effective Date:</b> ____________________, 20____",
    "re": "<b>RE:</b> Pilot Deployment of the GRADIS Privacy-Preserving Aerial "
          "Incident-Sensing System",
    "intro": "This letter of intent (the &ldquo;Letter of Intent&rdquo;) represents the "
             "basic terms for an agreement that shall be considered "
             f"{CB_OFF} binding &nbsp; {CB_ON} non-binding. After this Letter of Intent "
             "has been made, another formal agreement may be constructed to the benefit "
             "of the Parties.",
    "sections": [
        "<b>I. The Buyer:</b> ____________________________ (the &ldquo;Buyer&rdquo;) with "
        "a mailing address of ____________________________, City of ____________________, "
        "Province of ____________________, Republic of the Philippines. "
        "<i>[e.g., Local Government Unit public-safety office, PNP unit, or licensed "
        "private security agency.]</i>",

        "<b>II. The Seller:</b> GRADIS (the &ldquo;Seller&rdquo;), with a mailing "
        "address of ____________________________, City of ____________________, "
        "Republic of Korea.",

        "<b>III. The Transaction:</b> Buyer agrees to pay the Seller the amount of "
        "<b>Sixty Thousand US Dollars (USD 60,000)</b> (&ldquo;Purchase Price&rdquo;) in "
        "exchange for a <b>ninety (90)-day pilot deployment of the GRADIS system</b> "
        "(&ldquo;Goods/Services&rdquo;). The deployed production configuration is fixed as "
        "follows: <b>(i)</b> one (1) <b>DJI Matrice 4TD</b> aircraft (1,850 g; up to 54-min "
        "flight; IP55; 640&times;512 radiometric thermal with IR super-resolution; dual "
        "48 MP wide + tele cameras; 1,800 m laser rangefinder; omnidirectional binocular "
        "and 3D infrared obstacle sensing; full-color night mode); <b>(ii)</b> one (1) "
        "<b>DJI Dock 3</b> autonomous docking station (automated charging, launch and "
        "remote operation); <b>(iii)</b> one (1) <b>ground edge unit built on NVIDIA "
        "Jetson Orin (TensorRT)</b> that performs all anonymization on-device &mdash; "
        "YOLO-pose extraction of the GRADIS-25 25-joint skeleton and rendering of the "
        "rigged 3D mannequin (avatar); <b>the raw video never leaves this unit</b>, and "
        "only the anonymized mannequin frames are sent to a fast cloud vision-language "
        "model (Google Gemini Flash Lite class) for scene judgment; <b>(iv)</b> the "
        "GRADIS Core command dashboard "
        "with conditional 30-second incident-clip escalation; <b>(v)</b> video carried over "
        "the DJI Cloud API v2 live link (RTMP/RTSP via mediamtx) with an LTE/5G uplink for "
        "alerts; and <b>(vi)</b> on-site installation, operator training, and weekly "
        "analytics reporting over the pilot area agreed by the Parties. Throughout, <b>each "
        "person&rsquo;s pose is extracted and a rigged 3D mannequin (avatar) is overlaid in "
        "place of the real individual</b> &mdash; the actual face, body, and clothing are "
        "never shown, transmitted, or stored; only the anonymized mannequin layer is "
        "analyzed, which preserves detection accuracy while concealing identity. Flight "
        "operations maintain a geofence and a minimum 8-metre human stand-off.",

        "<b>IV. Payment:</b> Payment shall be made on a milestone basis: "
        "<b>(a) fifty percent (50%) as a mobilization payment</b> upon signing of the "
        "formal pilot agreement and <b>prior to deployment</b> of the GRADIS system; and "
        "<b>(b) the remaining fifty percent (50%) upon completion</b> of the 90-day pilot "
        "and delivery of the final analytics report. This balances the Seller&rsquo;s "
        "deployment costs with the Buyer&rsquo;s procurement practice.",

        "<b>V. Deposit:</b> "
        f"{CB_OFF} Deposit is NOT Required. &nbsp; {CB_ON} Deposit is Required &mdash; the "
        "50% mobilization payment under Section IV serves as the advance securing the "
        "pilot; no separate deposit applies.",

        "<b>VI. Financing:</b> "
        f"{CB_ON} NOT Conditional Upon Financing &mdash; this Letter of Intent is not "
        f"conditional on the Buyer&rsquo;s ability to obtain financing. &nbsp; {CB_OFF} "
        "Conditional Upon Financing.",

        "<b>VII. Currency:</b> All mentions of money or the usage of the &ldquo;$&rdquo; "
        "icon shall be known as referring to the US Dollar (USD).",

        "<b>VIII. Governing Law &amp; Compliance:</b> This Letter of Intent shall be "
        "governed under the laws of the Republic of the Philippines. The Parties intend "
        "that all flight operations be conducted in compliance with the Civil Aviation "
        "Authority of the Philippines (CAAP) Remotely Piloted Aircraft Systems rules "
        "under PCAR Part 11 (including Memorandum Circular No. 026-2025), and with "
        "applicable data-privacy law &mdash; Republic Act No. 10173 (Data Privacy Act of "
        "2012, enforced by the National Privacy Commission) and Republic Act No. 9995 "
        "(Anti-Photo and Video Voyeurism Act of 2009). GRADIS&rsquo;s mannequin-overlay "
        "design (no identifiable imagery is retained) is intended to keep no identifiable personal information within the scope "
        "of these laws.",
    ],
    "context_label": "<b>Background &amp; Intent of the Buyer.</b>",
    "context": "The Buyer operates in an environment where fixed CCTV coverage leaves "
               "significant gaps across streets, alleys, and barangay perimeters, and "
               "where the speed of incident detection materially affects public-safety "
               "outcomes. The Buyer expresses its good-faith intent to evaluate GRADIS as "
               "an active, airborne, privacy-by-design layer that detects incidents in "
               "real time without capturing the identity of any individual. This Letter of "
               "Intent records that intent and the Parties&rsquo; willingness to proceed "
               "toward a formal pilot agreement.",
    "buyer_hdr": "<b>BUYER</b> (Public-Safety Agency / Operator)",
    "seller_hdr": "<b>SELLER</b> (GRADIS)",
    "sig_line": "Signature ____________________________",
    "name_line": "Print Name ____________________________",
    "title_line": "Title / Office ____________________________",
    "title_line2": "Title ____________________________",
    "date_line": "Date ____________________________",
    "footer": "This Letter of Intent is non-binding except as the Parties may otherwise "
              "agree in writing, and is intended solely to record mutual interest in "
              "negotiating a formal pilot agreement. It is not a forecast of, or "
              "commitment to, any procurement.",
}

# ============================ SOUTH AFRICA (EN) ============================
ZA = {
    "doc_title": "GRADIS Letter of Intent - South Africa",
    "title": "LETTER OF INTENT",
    "subtitle": "(PILOT DEPLOYMENT &mdash; TRANSACTION)",
    "eff": "<b>Effective Date:</b> ____________________, 20____",
    "re": "<b>RE:</b> Pilot Deployment of the GRADIS Privacy-Preserving Aerial "
          "Incident-Sensing System",
    "intro": "This letter of intent (the &ldquo;Letter of Intent&rdquo;) represents the "
             "basic terms for an agreement that shall be considered "
             f"{CB_OFF} binding &nbsp; {CB_ON} non-binding. After this Letter of Intent "
             "has been made, another formal agreement may be constructed to the benefit "
             "of the Parties.",
    "sections": [
        "<b>I. The Buyer:</b> ____________________________ (the &ldquo;Buyer&rdquo;) with "
        "a mailing address of ____________________________, City of ____________________, "
        "Province of ____________________, Republic of South Africa. "
        "<i>[e.g., PSIRA-registered private security / armed-response company, estate or "
        "precinct association, or metro police department.]</i>",

        "<b>II. The Seller:</b> GRADIS (the &ldquo;Seller&rdquo;), with a mailing "
        "address of ____________________________, City of ____________________, "
        "Republic of Korea.",

        "<b>III. The Transaction:</b> Buyer agrees to pay the Seller the amount of "
        "<b>Sixty Thousand US Dollars (USD 60,000)</b> (&ldquo;Purchase Price&rdquo;) in "
        "exchange for a <b>ninety (90)-day pilot deployment of the GRADIS system</b> "
        "(&ldquo;Goods/Services&rdquo;). The deployed production configuration is fixed as "
        "follows: <b>(i)</b> one (1) <b>DJI Matrice 4TD</b> aircraft (1,850 g; up to 54-min "
        "flight; IP55; 640&times;512 radiometric thermal with IR super-resolution; dual "
        "48 MP wide + tele cameras; 1,800 m laser rangefinder; omnidirectional binocular "
        "and 3D infrared obstacle sensing; full-color night mode); <b>(ii)</b> one (1) "
        "<b>DJI Dock 3</b> autonomous docking station (automated charging, launch and "
        "remote operation); <b>(iii)</b> one (1) <b>ground edge unit built on NVIDIA "
        "Jetson Orin (TensorRT)</b> that performs all anonymization on-device &mdash; "
        "YOLO-pose extraction of the GRADIS-25 25-joint skeleton and rendering of the "
        "rigged 3D mannequin (avatar); <b>the raw video never leaves this unit</b>, and "
        "only the anonymized mannequin frames are sent to a fast cloud vision-language "
        "model (Google Gemini Flash Lite class) for scene judgment; <b>(iv)</b> the "
        "GRADIS Core command dashboard "
        "with conditional 30-second incident-clip escalation; <b>(v)</b> video carried over "
        "the DJI Cloud API v2 live link (RTMP/RTSP via mediamtx) with an LTE/5G uplink for "
        "alerts; and <b>(vi)</b> on-site installation, operator training, and weekly "
        "analytics reporting over the pilot area agreed by the Parties. Throughout, <b>each "
        "person&rsquo;s pose is extracted and a rigged 3D mannequin (avatar) is overlaid in "
        "place of the real individual</b> &mdash; the actual face, body, and clothing are "
        "never shown, transmitted, or stored; only the anonymized mannequin layer is "
        "analyzed, which preserves detection accuracy while concealing identity. Flight "
        "operations maintain a geofence and a minimum 8-metre human stand-off.",

        "<b>IV. Payment:</b> Payment shall be made on a milestone basis: "
        "<b>(a) fifty percent (50%) as a mobilization payment</b> upon signing of the "
        "formal pilot agreement and <b>prior to deployment</b> of the GRADIS system; and "
        "<b>(b) the remaining fifty percent (50%) upon completion</b> of the 90-day pilot "
        "and delivery of the final analytics report. This balances the Seller&rsquo;s "
        "deployment costs with the Buyer&rsquo;s procurement practice.",

        "<b>V. Deposit:</b> "
        f"{CB_OFF} Deposit is NOT Required. &nbsp; {CB_ON} Deposit is Required &mdash; the "
        "50% mobilization payment under Section IV serves as the advance securing the "
        "pilot; no separate deposit applies.",

        "<b>VI. Financing:</b> "
        f"{CB_ON} NOT Conditional Upon Financing &mdash; this Letter of Intent is not "
        f"conditional on the Buyer&rsquo;s ability to obtain financing. &nbsp; {CB_OFF} "
        "Conditional Upon Financing.",

        "<b>VII. Currency:</b> All mentions of money or the usage of the &ldquo;$&rdquo; "
        "icon shall be known as referring to the US Dollar (USD).",

        "<b>VIII. Governing Law &amp; Compliance:</b> This Letter of Intent shall be "
        "governed under the laws of the Republic of South Africa. The Parties intend that "
        "all flight operations be conducted in compliance with the South African Civil "
        "Aviation Authority (SACAA) Part 101 of the Civil Aviation Regulations, 2011 "
        "(including the Remote Operator Certificate requirement for commercial RPAS), the "
        "Private Security Industry Regulatory Authority (PSIRA) requirements applicable to "
        "security RPAS, and the Protection of Personal Information Act, 2013 (POPIA), "
        "enforced by the Information Regulator. GRADIS&rsquo;s mannequin-overlay design "
        "(no identifiable imagery is retained) is intended to minimize personal-information "
        "processing under POPIA.",
    ],
    "context_label": "<b>Background &amp; Intent of the Buyer.</b>",
    "context": "The Buyer operates in one of the world&rsquo;s most security-intensive "
               "environments, where private security and armed-response services carry a "
               "large share of frontline public safety and where rapid, accurate incident "
               "detection is decisive. The Buyer expresses its good-faith intent to "
               "evaluate GRADIS as an active, airborne, privacy-by-design layer that "
               "detects incidents in real time without capturing the identity of any "
               "individual &mdash; reducing both response time and false dispatches. This "
               "Letter of Intent records that intent and the Parties&rsquo; willingness to "
               "proceed toward a formal pilot agreement.",
    "buyer_hdr": "<b>BUYER</b> (Security Operator / Authority)",
    "seller_hdr": "<b>SELLER</b> (GRADIS)",
    "sig_line": "Signature ____________________________",
    "name_line": "Print Name ____________________________",
    "title_line": "Title / Capacity ____________________________",
    "title_line2": "Title ____________________________",
    "date_line": "Date ____________________________",
    "footer": "This Letter of Intent is non-binding except as the Parties may otherwise "
              "agree in writing, and is intended solely to record mutual interest in "
              "negotiating a formal pilot agreement. It is not a forecast of, or "
              "commitment to, any procurement.",
}

# ============================ MEXICO (ES) ============================
MX = {
    "doc_title": "GRADIS Carta de Intencion - Mexico",
    "title": "CARTA DE INTENCIÓN",
    "subtitle": "(DESPLIEGUE PILOTO &mdash; TRANSACCIÓN)",
    "eff": "<b>Fecha de Entrada en Vigor:</b> ____________________ de 20____",
    "re": "<b>ASUNTO:</b> Despliegue piloto del sistema GRADIS de detección aérea "
          "de incidentes con protección de la privacidad",
    "intro": "La presente carta de intención (la &ldquo;Carta de Intención&rdquo;) "
             "representa los términos básicos de un acuerdo que se "
             f"considerará {CB_OFF} vinculante &nbsp; {CB_ON} no vinculante. Una vez "
             "suscrita esta Carta de Intención, podrá elaborarse otro acuerdo "
             "formal en beneficio de las Partes.",
    "sections": [
        "<b>I. El Comprador:</b> ____________________________ (el "
        "&ldquo;Comprador&rdquo;), con domicilio en ____________________________, Ciudad "
        "de ____________________, Estado de ____________________, Estados Unidos "
        "Mexicanos. <i>[p. ej., Secretaría de Seguridad Ciudadana municipal, policía "
        "estatal o empresa de seguridad privada.]</i>",

        "<b>II. El Vendedor:</b> GRADIS (el &ldquo;Vendedor&rdquo;), con domicilio en "
        "____________________________, Ciudad de ____________________, República de "
        "Corea.",

        "<b>III. La Transacción:</b> El Comprador conviene en pagar al Vendedor la "
        "cantidad de <b>Sesenta Mil Dólares de los Estados Unidos (USD 60,000)</b> "
        "(&ldquo;Precio de Compra&rdquo;) a cambio de un <b>despliegue piloto de noventa "
        "(90) días del sistema GRADIS</b> (&ldquo;Bienes/Servicios&rdquo;). La "
        "configuración de producción desplegada queda fijada de la siguiente manera: "
        "<b>(i)</b> una (1) aeronave <b>DJI Matrice 4TD</b> (1,850 g; vuelo de hasta 54 "
        "min; IP55; térmica radiométrica 640&times;512 con superresolución IR; cámaras "
        "duales de 48 MP gran angular + teleobjetivo; telémetro láser de 1,800 m; "
        "detección de obstáculos binocular omnidireccional y 3D infrarroja; modo nocturno "
        "a todo color); <b>(ii)</b> una (1) estación de acoplamiento autónoma <b>DJI "
        "Dock 3</b> (carga, despegue y operación remota automatizados); <b>(iii)</b> una "
        "(1) <b>unidad de borde en tierra basada en NVIDIA Jetson Orin (TensorRT)</b> que "
        "realiza toda la anonimización en el dispositivo &mdash; extracción YOLO-pose del "
        "esqueleto GRADIS-25 de 25 articulaciones y renderizado del maniquí 3D (avatar) "
        "riggeado; <b>el video en bruto nunca sale de esta unidad</b>, y solo los "
        "fotogramas anonimizados del maniquí se envían a un modelo visión-lenguaje rápido "
        "en la nube (clase Google Gemini Flash Lite) para el juicio de la escena; "
        "<b>(iv)</b> el panel de mando GRADIS Core con "
        "escalamiento condicional de videoclips de 30 segundos; <b>(v)</b> video "
        "transportado por el enlace en vivo de la DJI Cloud API v2 (RTMP/RTSP vía "
        "mediamtx) con enlace ascendente LTE/5G para alertas; y <b>(vi)</b> instalación en "
        "sitio, capacitación de operadores e informes analíticos semanales sobre el área "
        "piloto acordada por las Partes. En todo momento, <b>se extrae la pose de cada "
        "persona y se superpone un maniquí 3D (avatar) en lugar de la persona real</b> "
        "&mdash; el rostro, el cuerpo y la ropa reales nunca se muestran, transmiten ni "
        "almacenan; solo se analiza la capa anonimizada del maniquí, lo que preserva la "
        "precisión de detección y oculta la identidad. Las operaciones de vuelo mantienen "
        "una geocerca y una distancia mínima de seguridad de 8 metros respecto a las "
        "personas.",

        "<b>IV. Pago:</b> El pago se realizará por etapas (hitos): "
        "<b>(a) cincuenta por ciento (50%) como pago de movilización</b> a la firma del "
        "acuerdo piloto formal y <b>antes del despliegue</b> del sistema GRADIS; y "
        "<b>(b) el cincuenta por ciento (50%) restante a la conclusión</b> del piloto de "
        "90 días y la entrega del informe analítico final. Esto equilibra los costos de "
        "despliegue del Vendedor con las prácticas de adquisición del Comprador.",

        "<b>V. Depósito:</b> "
        f"{CB_OFF} No se requiere depósito. &nbsp; {CB_ON} Se requiere depósito &mdash; "
        "el pago de movilización del 50% conforme a la Sección IV constituye el anticipo "
        "que garantiza el piloto; no aplica un depósito independiente.",

        "<b>VI. Financiamiento:</b> "
        f"{CB_ON} No condicionado a financiamiento &mdash; esta Carta de Intención no "
        f"está condicionada a la capacidad del Comprador para obtener financiamiento. "
        f"&nbsp; {CB_OFF} Condicionado a financiamiento.",

        "<b>VII. Moneda:</b> Toda mención de dinero o el uso del símbolo "
        "&ldquo;$&rdquo; se entenderá referida al Dólar de los Estados Unidos (USD).",

        "<b>VIII. Ley Aplicable y Cumplimiento:</b> Esta Carta de Intención se regirá "
        "por las leyes de los Estados Unidos Mexicanos. Las Partes pretenden que todas las "
        "operaciones de vuelo se realicen conforme a la normativa de la Agencia Federal de "
        "Aviación Civil (AFAC) en materia de RPAS y a la nueva Ley Federal de Protección "
        "de Datos Personales en Posesión de los Particulares (LFPDPPP), publicada el 20 de "
        "marzo de 2025 y supervisada por la Secretaría Anticorrupción y Buen Gobierno (en "
        "sustitución del extinto INAI), así como demás legislación aplicable. El diseño de "
        "GRADIS, basado en la superposición de un maniquí (sin conservar imágenes "
        "identificables), busca no conservar datos personales identificables dentro del "
        "ámbito de dichas leyes.",
    ],
    "context_label": "<b>Antecedentes e intención del Comprador.</b>",
    "context": "El Comprador opera en un entorno con alta incidencia delictiva, donde la "
               "cobertura de videovigilancia fija deja brechas importantes y donde la "
               "rapidez en la detección de incidentes incide directamente en la "
               "seguridad ciudadana. El Comprador manifiesta su intención de buena fe "
               "de evaluar GRADIS como una capa aérea, activa y diseñada con "
               "privacidad desde el origen, que detecta incidentes en tiempo real sin "
               "capturar la identidad de ninguna persona. Esta Carta de Intención hace "
               "constar dicha intención y la disposición de las Partes a avanzar "
               "hacia un acuerdo piloto formal.",
    "buyer_hdr": "<b>COMPRADOR</b> (Autoridad / Operador de Seguridad)",
    "seller_hdr": "<b>VENDEDOR</b> (GRADIS)",
    "sig_line": "Firma ____________________________",
    "name_line": "Nombre ____________________________",
    "title_line": "Cargo / Dependencia ____________________________",
    "title_line2": "Cargo ____________________________",
    "date_line": "Fecha ____________________________",
    "footer": "Esta Carta de Intención no es vinculante, salvo que las Partes acuerden "
              "lo contrario por escrito, y tiene por único objeto hacer constar el "
              "interés mutuo en negociar un acuerdo piloto formal. No constituye "
              "pronóstico ni compromiso de adquisición alguna.",
}

import os
base = os.path.dirname(os.path.abspath(__file__))
build(os.path.join(base, "GRADIS_LOI_Philippines_EN.pdf"), PH)
build(os.path.join(base, "GRADIS_LOI_SouthAfrica_EN.pdf"), ZA)
build(os.path.join(base, "GRADIS_LOI_Mexico_ES.pdf"), MX)
print("DONE")
