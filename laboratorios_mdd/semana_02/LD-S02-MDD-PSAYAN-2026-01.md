# LABORATORIO DIRIGIDO N.° 2 Vectores, Matrices y Operaciones Básicas Dataset: Titanic — Kaggle \| Semana 2 \| Unidad 1: Álgebra Lineal Aplicada

## I. Elemento De La Capacidad Terminal De La Semana

El estudiante aplica operaciones matriciales clave — producto matricial, matrices especiales y normalización— sobre datasets de negocio reales usando np.dot(), np.matmul() y np.linalg.norm() en Google Colab, interpretando cada resultado en contexto empresarial.

## II. Seguridad

- Prohibida la manipulación del hardware, conexiones eléctricas o de red sin supervisión.
- Prohibida la ingesta de alimentos y bebidas durante la sesión.
- Ubicar maletines y mochilas en el lugar destinado para tal fin.
- Dejar la mesa de trabajo y la silla limpias al finalizar.
- No compartir credenciales de acceso a Google Colab ni a plataformas del curso.
## III. Fundamento teórico

Producto matricial (np.dot / np.matmul): Si A tiene dimensiones m×n y B tiene dimensiones n×p, entonces A·B produce una matriz m×p. Cada elemento del resultado es la suma de los productos de una fila de A por una columna de B. Aplicación clave: calcular ingresos totales multiplicando cantidades por precios.

Matrices especiales: la matriz identidad (np.eye(n)) actúa como el '1' en la multiplicación matricial; la matriz diagonal (np.diag([...])) concentra valores en la diagonal principal y es útil para escalar filas o columnas de forma diferenciada.

Normalización Min-Max: transforma cada columna al rango [0,1] mediante la fórmula X_norm = (X - X.min()) / (X.max() - X.min()). Permite comparar variables con distintas escalas (e.g., ventas en unidades vs. ingresos en soles).

Revisión previa obligatoria: McKinney, W. (2022). Python for Data Analysis, Cap. 4 (NumPy Basics). VanderPlas, J. (2016). Python Data Science Handbook, Sección 2.

## IV. Normas empleadas

No aplica norma técnica específica. Se siguen las buenas prácticas de programación Python (PEP 8) y las políticas académicas del curso.

## V. Recursos

- Computadora con acceso a internet y cuenta Google activa.
- Google Colab — entorno de ejecución en la nube.
- Librerías NumPy y Matplotlib (disponibles por defecto en Colab).
- Sílabo del curso y material de semana 2 cargado en NotebookLM.
## VI. Metodología para el desarrollo de la tarea

- El desarrollo del laboratorio es individual.
- Cada estudiante trabaja en su propio notebook de Google Colab.
- El enlace del notebook ejecutado se adjunta en el foro semanal como evidencia de portafolio.
## VII. PROCEDIMIENTO

### CASO: SportGear Perú — Análisis de Ingresos por Sucursal

SportGear Perú es una cadena de tiendas de artículos deportivos con sucursales en Lima, Arequipa, Cusco y Piura. Sus tres productos estrella son zapatillas, camisetas y shorts. El área de ventas registra mensualmente las cantidades vendidas por producto y sucursal, mientras que finanzas mantiene una tabla de precio unitario y costo unitario por producto. El gerente general necesita calcular: (1) los ingresos brutos por sucursal, (2) el margen de ganancia por sucursal y (3) comparar el rendimiento de sucursales tras normalizar los datos.

**Tabla de cantidades vendidas (unidades) — Mayo 2026:**

| Producto | Lima | Arequipa | Cusco | Piura |
| --- | --- | --- | --- | --- |
| Zapatillas | 320 | 180 | 110 | 95 |
| Camisetas | 850 | 410 | 260 | 220 |
| Shorts | 640 | 300 | 195 | 170 |

**Tabla de precio y costo unitario (S/.):**

| Producto | Precio unit. (S/.) | Costo unit. (S/.) |
| --- | --- | --- |
| Zapatillas | 250 | 140 |
| Camisetas | 45 | 18 |
| Shorts | 35 | 12 |

Usted asumirá el rol del analista de datos de SportGear y resolverá las preguntas del gerente usando operaciones matriciales en Python.

### Actividad 1: Revisión de conceptos

Complete la tabla antes de iniciar el código.

| Concepto | ¿Qué hace? / ¿Cuándo se usa? |
| --- | --- |
| np.dot(A, B) |  |
| np.matmul(A, B) |  |
| np.eye(n) |  |
| np.diag([a, b, c]) |  |
| Normalización Min-Max |  |
| Condición de conformabilidad (m×n · n×p) |  |

### Actividad 2: Desarrollo en Google Colab

Cree el notebook LD02_SportGear_[SuApellido].ipynb y ejecute los pasos siguientes.

#### Paso 1 — Definir matrices de datos

```python
import numpy as np
import matplotlib.pyplot as plt
# Matriz de cantidades: 3 productos × 4 sucursales
Q = np.array([
    [320, 180, 110,  95],   # Zapatillas
    [850, 410, 260, 220],   # Camisetas
    [640, 300, 195, 170],   # Shorts
])
# Matriz de precios y costos: 3 productos × 2 columnas
PC = np.array([
    [250, 140],   # Zapatillas: precio, costo
    [ 45,  18],   # Camisetas
    [ 35,  12],   # Shorts
])
sucursales = ['Lima','Arequipa','Cusco','Piura']
productos  = ['Zapatillas','Camisetas','Shorts']
print('Q shape:', Q.shape, '  PC shape:', PC.shape)
```

#### Paso 2 — Ingresos y costos totales por sucursal (producto matricial)

```python
# Para cada sucursal: ingreso = sum(precio_i × cantidad_i) para todos los productos
# PC.T tiene shape (2, 3); Q tiene shape (3, 4) → resultado (2, 4)
resultado = np.matmul(PC.T, Q)   # shape: 2 × 4
ingresos = resultado[0]           # fila 0: ingresos brutos
costos   = resultado[1]           # fila 1: costos totales
margen   = ingresos - costos      # ganancia por sucursal
print('\nIngresos brutos por sucursal (S/.):')
for s, ing in zip(sucursales, ingresos):
    print(f'  {s:10s}: S/. {ing:>10,.2f}')
print('\nMargen de ganancia por sucursal (S/.):')
for s, m in zip(sucursales, margen):
    print(f'  {s:10s}: S/. {m:>10,.2f}')
```

#### Paso 3 — Matriz diagonal para ajuste de precios diferenciado

```python
# El gerente decide subir precios: Zapatillas +10%, Camisetas +5%, Shorts +8%
D = np.diag([1.10, 1.05, 1.08])   # Matriz diagonal de factores
print('Matriz diagonal de ajuste:\n', D)
Q_ajustado = np.dot(D, PC)         # Nuevos precios y costos
print('\nPrecios y costos ajustados:')
for p, fila in zip(productos, Q_ajustado):
    print(f'  {p:12s}  Precio: S/. {fila[0]:6.2f}  Costo: S/. {fila[1]:6.2f}')
```

#### Paso 4 — Normalización Min-Max de ingresos

```python
# Normalizar ingresos por sucursal al rango [0, 1]
ing_min = ingresos.min()
ing_max = ingresos.max()
ing_norm = (ingresos - ing_min) / (ing_max - ing_min)
print('Ingresos normalizados (0 = peor, 1 = mejor sucursal):')
for s, n in zip(sucursales, ing_norm):
    print(f'  {s:10s}: {n:.4f}')
```

#### Paso 5 — Visualización comparativa

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colores = ['#1F4E79','#2E74B5','#9DC3E6','#BDD7EE']
# Gráfico 1: Ingresos vs Costos
x = np.arange(len(sucursales))
axes[0].bar(x - 0.2, ingresos/1000, 0.35, label='Ingresos', color='#1F4E79')
axes[0].bar(x + 0.2, costos/1000,   0.35, label='Costos',   color='#9DC3E6')
axes[0].set_title('Ingresos vs Costos (miles S/.)'); axes[0].set_xticks(x); axes[0].set_xticklabels(sucursales); axes[0].legend()
# Gráfico 2: Ingresos normalizados
axes[1].bar(sucursales, ing_norm, color=colores)
axes[1].set_title('Rendimiento normalizado por sucursal (0-1)')
axes[1].set_ylabel('Score normalizado')
for i, v in enumerate(ing_norm):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout(); plt.show()
```

### Actividad 3: Análisis e interpretación de resultados

Responda con datos concretos de su ejecución en Colab.

#### Pregunta 1.  ¿Qué dimensiones tienen las matrices Q y PC.T antes del producto matricial? ¿Por qué la condición de conformabilidad se cumple? Complete la tabla:

| Operación | Dimensión A | Dimensión B | Dimensión resultado |
| --- | --- | --- | --- |
| PC.T · Q |  |  |  |
| D · PC |  |  |  |

#### Pregunta 2.  ¿Cuál es la sucursal con mayor ingreso bruto y cuál con mayor margen de ganancia? ¿Coinciden? Explique por qué pueden diferir y qué implicancia tiene para el gerente al tomar decisiones.

#### Pregunta 3.  Tras aplicar la matriz diagonal de ajuste de precios, ¿qué producto recibió el mayor incremento en precio absoluto (S/.)? ¿Le parece que un ajuste diferenciado por producto es más justo que uno uniforme? Sustente.

#### Pregunta 4.  En la normalización Min-Max, Piura obtuvo el score más bajo. ¿Eso significa que es una sucursal poco rentable o solo que tiene el menor volumen? ¿Qué otros indicadores revisaría para tomar una decisión sobre Piura?

★ RETO — Pregunta 5.  El gerente desea saber el ingreso por producto (no por sucursal). Transponga el resultado del Paso 2 y calcule cuánto aportó cada producto (zapatillas, camisetas, shorts) al ingreso total nacional. Exprese los resultados como porcentaje del total.

## Conclusiones.

Instrucción: Liste al menos tres conclusiones técnicas y de aprendizaje. Redacte en sus propias palabras.

- 1.
- 2.
- 3.
## Material Complementario.

- 1.  McKinney, W. (2022). Python for Data Analysis, Cap. 4. O'Reilly Media.
- 2.  VanderPlas, J. (2016). Python Data Science Handbook, Sección 2. https://jakevdp.github.io/PythonDataScienceHandbook/
- 3.  NumPy Documentation — numpy.matmul. https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
## Criterios De Evaluación — Rúbrica Holística

| Curso | Matemática Aplicada a la Ciencia de Datos | Semana | 2 |
| --- | --- | --- | --- |
| Actividad | LD-S02 — Operaciones Matriciales | Semestre | 2026-I |
| Nombre del alumno |  | Fecha |  |
| Docente |  | Sección | PSAYAN |

| Criterio | Excelente (4) | Bueno (3) | Requiere mejora (2) | No aceptable (1) | Puntaje |
| --- | --- | --- | --- | --- | --- |
| 1. Producto matricial (np.matmul) | Implementa correctamente PC.T·Q, obtiene ingresos y costos sin errores, interpreta shape. | Implementa el producto; pequeño error en indexación de filas pero resultado coherente. | Intenta el producto pero confunde dimensiones o indexa incorrectamente. | No implementa el producto matricial o produce error no resuelto. |  |
| 2. Matriz diagonal y ajuste | Aplica np.diag correctamente, calcula nuevos precios e interpreta el efecto diferenciado. | Aplica np.diag con un error menor; interpreta parcialmente. | Intenta la diagonal con errores que afectan el resultado. | No aplica la matriz diagonal. |  |
| 3. Normalización Min-Max | Normaliza correctamente, interpreta el score de cada sucursal. | Normaliza con fórmula correcta; interpretación incompleta. | Aplica fórmula con error menor; resultado parcialmente incorrecto. | No aplica normalización o la fórmula es incorrecta. |  |
| 4. Visualización | Genera ambos gráficos con título, etiquetas y leyenda correctas. | Genera los gráficos; faltan algunos elementos de presentación. | Genera al menos un gráfico con errores menores. | No genera los gráficos o están completamente incorrectos. |  |
| 5. Interpretación + Reto | Responde las 4 preguntas con datos y completa el reto de aporte por producto. | Responde 3 preguntas correctamente e intenta el reto. | Responde 2 preguntas; no intenta el reto. | Responde menos de 2 preguntas sin sustento en datos. |  |
| TOTAL |  |  |  |  | /20 |

| Acciones a cumplir | Descuento |
| --- | --- |
| Puntualidad y dedicación durante la sesión | -1 pt |
| Notebook entregado sin outputs ejecutados | -1 pt |
| Conclusiones: ortografía, redacción o ausencia de argumentos | -1 pt |

Comentarios del docente:  ________________________________________________________________________________________________________________________________________________

## _______________________________________________________________________________________________________
