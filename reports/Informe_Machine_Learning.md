# Informe Técnico — Machine Learning

**Profesora:** Jazna Meza

**Integrantes:**

- Diego Améstica  
- Gino Andrades  
- Agustín Cáceres

---

## Índice

| Tema | Página en PDF |
| :---- | ----: |
| 01\. Resumen ejecutivo y objetivos | 3 |
| 02\. Datos, EDA y CRISP-DM | 4 |
| 03\. Clasificación y optimización | 5 |
| 04\. ROC e impacto de la decisión | 6 |
| 05\. Regresión y generalización | 7 |
| 06\. K-Means y acciones por clúster | 8 |
| 07\. Conclusiones y recomendaciones | 9 |

---

## 01\. Resumen ejecutivo y objetivos

El proyecto estudia la popularidad de canciones de Spotify mediante tres enfoques complementarios: **clasificación** en Baja, Media y Alta; **regresión** del puntaje de 0 a 100; y **K-Means** para descubrir perfiles acústicos. Su propósito es apoyar la revisión editorial y el diseño de playlists, no sustituir decisiones humanas.

### Pregunta de negocio

¿Cómo priorizar canciones potencialmente populares y organizar un catálogo diverso usando evidencia cuantitativa, sin confundir popularidad con calidad artística?

### Objetivos específicos

Comparar modelos base y optimizados; justificar el uso de ensambles; evaluar la generalización mediante entrenamiento, validación cruzada y prueba; y proponer acciones por segmento con métricas verificables.

### Indicadores clave

| Enfoque | KPI técnico | Uso propuesto |
| :---- | :---- | :---- |
| Clasificación | F1 macro y recall de Alta | Priorizar revisión de canciones |
| Regresión | MAE y R² | Estimar popularidad numérica |
| Segmentación | Silhouette y perfiles | Diseñar playlists temáticas |

Los KPI comerciales —guardados, saltos, finalización, exposición de artistas y valor incremental neto— se proponen para pilotos futuros; no fueron medidos en el dataset.

## 02\. Datos, EDA y CRISP-DM

| Medida | Resultado |
| :---- | :---- |
| Registros originales | 114.000 filas · 21 columnas |
| Pistas únicas | 89.741 identificadas por `track_id` |
| Géneros | 114 |
| Registros adicionales por ID | 24.259; no todos son copias idénticas |

### Preparación y calidad

Se examinaron nulos, repeticiones, distribuciones y valores extremos. La consolidación por `track_id` evita que una misma canción se repita en entrenamiento y prueba, pero puede perder géneros secundarios. Se conservaron valores acústicos extremos plausibles y se excluyó la variable objetivo de los predictores.

### Transformaciones

**Clasificación:** categorías Baja (0–24), Media (25–49) y Alta (50–100), Target Encoding de género y otras codificaciones según el pipeline. **Regresión:** imputación y One-Hot. **K-Means:** nueve atributos acústicos imputados y estandarizados para comparar distancias.

### Metodología

CRISP-DM: comprensión del negocio → comprensión de datos → preparación → modelado → evaluación → propuesta de despliegue. En modelos supervisados se reservó 20 % para prueba; los transformadores se ajustaron con entrenamiento.

## 03\. Clasificación y optimización

Se compararon árboles de decisión y Random Forest en versiones base y optimizadas. Se seleccionó **Random Forest optimizado** por su mayor F1 macro en el conjunto de prueba.

| Modelo | F1 macro · prueba |
| :---- | ----: |
| Decision Tree base | ≈ 0,611 |
| Decision Tree optimizado | ≈ 0,644 |
| Random Forest base | 0,6977 |
| **Random Forest optimizado** | **0,7007** |

### Generalización

| Métrica RF optimizado | Valor |
| :---- | ----: |
| F1 macro entrenamiento | 0,9504 |
| F1 macro validación cruzada | 0,6885 |
| F1 macro prueba | 0,7007 |

La diferencia entre entrenamiento y prueba evidencia sobreajuste. La validación cruzada y el test muestran un rendimiento más conservador. La optimización mejora moderadamente al Random Forest base; no elimina la brecha de generalización.

### Interpretación

El ensamble combina múltiples árboles para reducir la inestabilidad de un árbol individual. La métrica macro otorga el mismo peso a las tres clases, mientras que el recall de Alta permite medir cuántas canciones populares se detectan.

## 04\. ROC e impacto de la decisión

En el modelo multiclase, la regla original es **`argmax`**: se selecciona la clase con mayor probabilidad, sin umbrales fijos para Baja, Media ni Alta. En validación se seleccionó **0,295 para Alta** con el objetivo de reducir omisiones, limitando el aumento de falsas alertas.

| Métrica en prueba | Argmax | ROC 0,295 |
| :---- | ----: | ----: |
| Verdaderos positivos · Alta | 2.564 | 3.281 |
| Falsos negativos · Alta | 1.678 | 961 |
| Falsos positivos · Alta | 1.404 | 2.729 |
| Recall Alta | 60,44 % | 77,35 % |
| F1 macro | 0,7007 | 0,6841 |

**Balance:** 717 canciones altas adicionales detectadas y 1.325 falsas alertas adicionales. La detección mejora, pero el F1 macro global disminuye.

### Simulación económica ilustrativa

Supuestos: **$50.000CLP**devalorporverdaderopositivoy**$20.000 CLP** de costo por cada canción seleccionada (VP \+ FP).

| Política | Neto hipotético |
| :---- | ----: |
| RF base | \$48,01 millones |
| RF optimizado · argmax | \$48,84 millones |
| RF optimizado · ROC 0,295 | \$43,85 millones |

Los montos son ficticios, no ganancias reales de Spotify. Bajo estos supuestos, `argmax` supera al umbral ROC en **\$4,99 millones netos**.

## 05\. Regresión y generalización

Se modeló el puntaje continuo de popularidad entre 0 y 100\. El criterio principal es MAE (error absoluto medio); se complementa con R², que expresa la proporción de variación explicada en los datos evaluados.

| Modelo | MAE · prueba |
| :---- | ----: |
| Decision Tree base | 15,81 |
| Decision Tree optimizado | 11,82 |
| HistGradientBoosting base | 11,78 |
| **HistGradientBoosting optimizado** | **11,15** |

### Modelo elegido

| Indicador | Resultado |
| :---- | ----: |
| MAE en prueba | 11,15 puntos |
| R² en prueba | 0,4030 |
| MAE validación cruzada | 11,33 ± 0,07 puntos |

HistGradientBoosting construye árboles secuencialmente para corregir errores anteriores. Reduce el MAE frente al árbol individual, aunque R² \= 0,4030 refleja capacidad predictiva parcial, no un 40 % de canciones acertadas.

### Limitaciones

El error es mayor en canciones de popularidad alta y existen indicios de subestimación en ese tramo. Por ello, el puntaje sirve como apoyo a la revisión, no como garantía de éxito comercial.

## 06\. K-Means y acciones por clúster

Se aplicó K-Means sobre nueve atributos acústicos estandarizados, sin utilizar la popularidad para formar los grupos. Se eligió K=4 para disponer de perfiles comerciales interpretables.

| Selección | Silhouette |
| :---- | ----: |
| K=2 | 0,2456 |
| K=4 | 0,1613 |

K=2 tiene mejor separación geométrica. K=4 ofrece más detalle para propuestas de playlists, aunque los grupos se solapan y deben validarse con datos de escucha.

### Perfiles y pilotos

| Grupo | Tamaño | Acción sugerida |
| :---- | :---- | :---- |
| 0 · Acústico y tranquilo | 13.870 · 15,46 % | Listas de calma y estudio |
| 1 · En vivo y hablado | 7.277 · 8,11 % | Conciertos, humor y voces |
| 2 · Rápido e intenso | 29.505 · 32,88 % | Ejercicio y energía |
| 3 · Bailable y positivo | 39.089 · 43,56 % | Fiesta y buen ánimo |

Estos grupos describen canciones, no personas. Las acciones son hipótesis editoriales; su efecto debe probarse mediante experimentos A/B y métricas de experiencia.

## 07\. Conclusiones y recomendaciones

### Selección analítica

**Clasificación:** Random Forest optimizado con `argmax` como referencia operativa; evaluar el umbral 0,295 solo si recuperar más canciones altas compensa las falsas alertas. **Regresión:** HistGradientBoosting optimizado para puntajes orientativos. **Segmentación:** K-Means con cuatro perfiles para pilotos de playlists.

### Plan de implementación

1. Curar muestras por clúster y revisar manualmente candidatas.  
2. Ejecutar pruebas A/B frente a listas actuales.  
3. Medir guardados, saltos, finalización, diversidad de artistas y costos reales.  
4. Escalar solo si se demuestra valor incremental.

### Sesgos, ética y límites

La popularidad histórica puede reflejar exposición previa y no calidad artística. La deduplicación por ID puede reducir la representación de géneros secundarios. Se recomienda auditar resultados por género y artista, evitar excluir automáticamente música de nicho y respetar privacidad y licencias.