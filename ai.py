import json
from google import genai

# Se prueban modelos Flash en orden de prioridad. Algunos modelos antiguos
# pueden estar retirados o no habilitados en todos los proyectos; el fallback
# permite continuar con el siguiente identificador disponible.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def generate_economic_analysis(api_key: str, analysis_type: str,
                               scenario: dict, summary, correlations,
                               extra_question: str = "") -> tuple[str, str]:
    if not api_key:
        raise ValueError("Falta configurar GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)

    if analysis_type == "individual":
        title = "ANÁLISIS DEL IMPACTO INDIVIDUAL DE COMMODITIES"
        instructions = """
        Analizá el impacto por complejo, identificá las mayores pérdidas y
        ganancias, explicá el efecto agregado y evaluá qué subas compensan
        parcialmente las caídas. No describas contagio: los shocks son
        elegidos individualmente por el usuario.
        """
    else:
        title = "ANÁLISIS DEL SHOCK SISTÉMICO Y CONTAGIO"
        instructions = """
        Identificá el complejo disparador, diferenciá impacto directo e
        indirecto, explicá cómo las betas históricas propagan el shock y
        compará los bloques Agro y Energía/Minería. Las betas son asociaciones
        históricas, no relaciones causales.
        """

    scenario_payload = {}
    for key, value in scenario.items():
        if hasattr(value, "to_dict"):
            scenario_payload[key] = value.to_dict()
        else:
            scenario_payload[key] = value

    prompt = f"""
    Actuá como analista económico especializado en comercio exterior argentino
    y mercados internacionales de commodities.

    {title}

    {instructions}

    Organizá la respuesta en:
    1. Lectura del escenario.
    2. Impacto sobre la canasta exportadora.
    3. Riesgos, compensaciones o contagio, según corresponda.
    4. Conclusión económica.

    RESUMEN HISTÓRICO:
    {summary.to_string()}

    MATRIZ DE CORRELACIONES:
    {correlations.to_string()}

    ESCENARIO SELECCIONADO:
    {json.dumps(scenario_payload, ensure_ascii=False, indent=2, default=str)}

    LIMITACIONES:
    - Los valores exportados corresponden al primer semestre de 2026.
    - Las cantidades exportadas se mantienen constantes.
    - Los precios son referencias internacionales, no precios FOB efectivos.
    - Las correlaciones y betas no prueban causalidad.
    - No infieras efectos fiscales, cambiarios ni sobre reservas: no se modelan.
    - No inventes cifras ni presentes escenarios como pronósticos reales.
    - Expresá los impactos monetarios en millones de USD.

    Escribí un informe claro y profesional, de hasta 350 palabras.
    """

    if extra_question.strip():
        prompt += f"\n\nPREGUNTA ADICIONAL DEL USUARIO:\n{extra_question}\nRespondela al final sin exceder los datos del modelo."

    errors = []
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response.text:
                return response.text, model_name
            errors.append(f"{model_name}: respuesta vacía.")
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "Ningún modelo Gemini pudo responder. Detalles:\n" + "\n".join(errors)
    )
