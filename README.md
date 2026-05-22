# Prediccion de paradas de maquina en operacion minera

## Objetivo

Construir un modelo de clasificacion para anticipar el riesgo de parada no programada de equipos mineros a partir de variables operativas y de condicion.

## Tipo de data

La data es sintetica propia. Fue generada con reglas reproducibles para simular una operacion minera con turnos, camiones/equipos, ciclos, carga, pendiente de ruta, temperatura, vibracion, presion hidraulica, consumo, mantenimiento y eventos de parada.

No representa una mina real ni informacion confidencial. Su funcion es demostrar criterio analitico, modelamiento y capacidad pedagogica.

## Pregunta de negocio

Que equipos tienen mayor riesgo de parada en las proximas horas y que factores explican ese riesgo?

## Variables principales

- `equipo_id`
- `turno`
- `horas_operacion_acumuladas`
- `toneladas_transportadas`
- `tiempo_ciclo_min`
- `pendiente_promedio_pct`
- `temperatura_motor_c`
- `vibracion_mm_s`
- `presion_hidraulica_bar`
- `combustible_l_h`
- `horas_desde_mantenimiento`
- `parada_no_programada`

## Como ejecutar

1. Ejecutar `src/generar_data_sintetica.py` para regenerar la data si se desea.
2. Abrir `notebooks/02_prediccion_paradas_maquina.ipynb`.
3. Ejecutar las celdas en orden.

## Resultado esperado

El notebook entrega:

- exploracion de paradas por turno y condicion,
- modelo base,
- modelo Random Forest,
- matriz de confusion y metricas,
- importancia de variables,
- lectura operativa para priorizar mantenimiento.
