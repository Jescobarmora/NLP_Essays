# NLP Ensayos

Este repositorio contiene la solución al **Caso de Estudio 2 - Ensayos**, basado en las pruebas PISA. A continuación, se describen las tareas realizadas y cómo reproducir los resultados.

---

## **Descripción del Problema**
En las pruebas PISA, los estudiantes escribieron un ensayo en uno de los 8 temas propuestos, los cuales están disponibles en la última columna del archivo `training_set_rel3.tsv`. Los objetivos del proyecto son:

1. **Procesamiento y tratamiento de datos:** Preparar los datos de cada ensayo para su análisis.
2. **Nubes de palabras:** Crear nubes de palabras para cada tema, con interpretaciones correspondientes.
3. **Implementación de metodologías:**
   - Aplicar modelos basados en LDA (con y sin TF-IDF), BERT y FastText para identificar los temas.
   - Evaluar las metodologías usando métricas como **accuracy** y **ROC AUC**.
   - Determinar qué método identifica mejor los temas.
   - Si ningún método recupera los 8 temas, proponer estrategias para mejorar el ajuste.
4. **Pipeline y pruebas automatizadas:**
   - Desarrollar un pipeline para automatizar los procedimientos anteriores.
   - Implementar pruebas unitarias y de integración en **Visual Studio Code**, que incluyan:
     - Validación del mejor modelo seleccionado.
     - Generación de una tabla en CSV con el área bajo la curva (ROC AUC) para cada clase y metodología, ordenada de mayor a menor rendimiento.

---

## **Ejecución del Código**

Main:
`python -m src.main`

Pruebas de integración:
`python -m unittest discover -s tests`
