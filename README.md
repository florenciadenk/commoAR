# CommoAR AI — Aplicación web

Aplicación web en Streamlit construida a partir del notebook CommoAR AI.

## Contenido

- `app.py`: interfaz web con Dashboard, Simuladores, Analista IA y Fuentes y metodología.
- `core.py`: descarga de precios, índice CommoAR, indicadores, Monte Carlo y shocks.
- `ai.py`: análisis económico con Gemini y modelos Flash alternativos.
- `requirements.txt`: dependencias.
- `.streamlit/secrets.toml.example`: ejemplo de configuración local de la API Key.

## Ejecutar localmente

1. Instalar Python 3.10 o superior.
2. Abrir una terminal en esta carpeta.
3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Crear `.streamlit/secrets.toml` a partir de `.streamlit/secrets.toml.example` y completar la clave de Gemini.
5. Ejecutar:

   ```bash
   streamlit run app.py
   ```

## Publicar en Streamlit Community Cloud

1. Crear un repositorio de GitHub con estos archivos.
2. En Streamlit Community Cloud, crear una nueva app y seleccionar `app.py`.
3. En la configuración de Secrets, agregar:

   ```toml
   GEMINI_API_KEY = "TU_API_KEY"
   ```

4. No subir nunca la API Key al repositorio.

## Decisiones metodológicas y notas

- Se conservan los cinco complejos y los valores de exportación del primer semestre de 2026 del notebook.
- Los precios de Yahoo Finance son referencias internacionales de futuros, no precios FOB efectivos argentinos.
- El oro es una referencia para el complejo de oro y plata; WTI es una referencia para petróleo.
- Las ponderaciones se mantienen fijas durante todo el período.
- Para Monte Carlo, esta versión utiliza los retornos logarítmicos diarios del propio índice CommoAR. Esto hace consistente la simulación con el índice ponderado de niveles de precios. El notebook original utilizaba una suma ponderada de retornos logarítmicos individuales, que es una aproximación distinta.
- La aplicación usa 5.000 simulaciones para dar más estabilidad al percentil 5 que las 1.000 simulaciones iniciales del notebook.
- Las betas describen asociaciones históricas y no causalidad.
- Las cantidades exportadas se mantienen constantes en los shocks.

## Modelos Gemini

`ai.py` intenta los modelos Flash de la lista en orden. Si un modelo no está disponible o falla, prueba el siguiente. La disponibilidad concreta depende del acceso y cuotas de la API de Google.
