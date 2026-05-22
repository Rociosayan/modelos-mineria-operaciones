from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_REGISTROS = 7200
N_EQUIPOS = 24


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generar_data():
    rng = np.random.default_rng(RANDOM_STATE)

    equipos = [f"CA-{i:03d}" for i in range(1, N_EQUIPOS + 1)]
    tipos = rng.choice(["camion_haul", "cargador_frontal", "perforadora"], size=N_EQUIPOS, p=[0.65, 0.2, 0.15])
    antiguedad = rng.integers(1, 12, size=N_EQUIPOS)
    criticidad = rng.choice(["alta", "media", "baja"], size=N_EQUIPOS, p=[0.35, 0.45, 0.20])

    catalogo = pd.DataFrame(
        {
            "equipo_id": equipos,
            "tipo_equipo": tipos,
            "antiguedad_anios": antiguedad,
            "criticidad": criticidad,
        }
    )

    fechas = pd.date_range("2026-01-01", periods=N_REGISTROS, freq="h")
    data = pd.DataFrame(
        {
            "fecha_hora": fechas,
            "equipo_id": rng.choice(equipos, size=N_REGISTROS),
            "turno": rng.choice(["dia", "noche"], size=N_REGISTROS, p=[0.52, 0.48]),
            "zona_operacion": rng.choice(["tajo_norte", "tajo_sur", "botadero", "chancadora"], size=N_REGISTROS),
            "clima": rng.choice(["normal", "lluvia", "polvo_alto"], size=N_REGISTROS, p=[0.68, 0.17, 0.15]),
        }
    )
    data = data.merge(catalogo, on="equipo_id", how="left")

    es_camion = data["tipo_equipo"].eq("camion_haul").astype(int)
    es_noche = data["turno"].eq("noche").astype(int)
    lluvia = data["clima"].eq("lluvia").astype(int)
    polvo = data["clima"].eq("polvo_alto").astype(int)
    alta_criticidad = data["criticidad"].eq("alta").astype(int)

    data["horas_operacion_acumuladas"] = (
        rng.normal(4200, 1200, N_REGISTROS)
        + data["antiguedad_anios"] * rng.normal(380, 60, N_REGISTROS)
    ).clip(300, 12500).round(1)

    data["pendiente_promedio_pct"] = (
        rng.normal(6.5, 2.2, N_REGISTROS)
        + data["zona_operacion"].eq("tajo_norte").astype(int) * 1.2
        + lluvia * 0.7
    ).clip(0.5, 14).round(2)

    data["toneladas_transportadas"] = (
        rng.normal(185, 38, N_REGISTROS)
        + es_camion * 40
        - lluvia * 18
        - data["zona_operacion"].eq("chancadora").astype(int) * 25
    ).clip(30, 320).round(1)

    data["tiempo_ciclo_min"] = (
        rng.normal(32, 6, N_REGISTROS)
        + data["pendiente_promedio_pct"] * 0.9
        + lluvia * 4.5
        + es_noche * 1.8
        - es_camion * 2.0
    ).clip(12, 75).round(1)

    data["horas_desde_mantenimiento"] = (
        rng.gamma(shape=4.5, scale=38, size=N_REGISTROS)
        + data["antiguedad_anios"] * 4
    ).clip(1, 520).round(1)

    desgaste = data["horas_desde_mantenimiento"] / 120 + data["antiguedad_anios"] / 8

    data["temperatura_motor_c"] = (
        rng.normal(84, 6.5, N_REGISTROS)
        + data["pendiente_promedio_pct"] * 1.1
        + desgaste * 2.3
        + lluvia * 1.5
        + polvo * 2.0
    ).clip(65, 125).round(1)

    data["vibracion_mm_s"] = (
        rng.normal(3.1, 0.8, N_REGISTROS)
        + desgaste * 0.55
        + data["zona_operacion"].eq("botadero").astype(int) * 0.35
        + polvo * 0.25
    ).clip(0.7, 9.5).round(2)

    data["presion_hidraulica_bar"] = (
        rng.normal(235, 18, N_REGISTROS)
        - desgaste * 6.5
        - data["temperatura_motor_c"].sub(90).clip(lower=0) * 0.8
        + rng.normal(0, 7, N_REGISTROS)
    ).clip(145, 285).round(1)

    data["combustible_l_h"] = (
        rng.normal(58, 9, N_REGISTROS)
        + data["toneladas_transportadas"] * 0.055
        + data["pendiente_promedio_pct"] * 1.3
        + data["tiempo_ciclo_min"] * 0.18
        + es_noche * 1.5
    ).clip(25, 105).round(1)

    data["alertas_previas_24h"] = rng.poisson(
        lam=(0.7 + desgaste * 0.35 + data["vibracion_mm_s"] * 0.08 + alta_criticidad * 0.2),
        size=N_REGISTROS,
    ).clip(0, 8)

    riesgo_lineal = (
        -5.3
        + data["temperatura_motor_c"].sub(95).clip(lower=0) * 0.075
        + data["vibracion_mm_s"].sub(4.5).clip(lower=0) * 0.55
        + data["horas_desde_mantenimiento"].sub(160).clip(lower=0) * 0.007
        + data["tiempo_ciclo_min"].sub(42).clip(lower=0) * 0.05
        + data["presion_hidraulica_bar"].rsub(205).clip(lower=0) * 0.025
        + es_noche * 0.35
        + lluvia * 0.30
        + alta_criticidad * 0.28
        + rng.normal(0, 0.35, N_REGISTROS)
    )
    prob_parada = sigmoid(riesgo_lineal)
    data["riesgo_parada_estimado"] = prob_parada.round(4)
    data["parada_no_programada"] = rng.binomial(1, prob_parada)

    condiciones = [
        data["temperatura_motor_c"].gt(104),
        data["vibracion_mm_s"].gt(5.7),
        data["presion_hidraulica_bar"].lt(195),
        data["horas_desde_mantenimiento"].gt(220),
    ]
    causas = ["sobretemperatura", "vibracion_alta", "baja_presion_hidraulica", "mantenimiento_vencido"]
    data["causa_probable"] = np.select(condiciones, causas, default="operacion_normal")
    data.loc[data["parada_no_programada"].eq(0), "causa_probable"] = "sin_parada"

    columnas = [
        "fecha_hora",
        "equipo_id",
        "tipo_equipo",
        "antiguedad_anios",
        "criticidad",
        "turno",
        "zona_operacion",
        "clima",
        "horas_operacion_acumuladas",
        "toneladas_transportadas",
        "tiempo_ciclo_min",
        "pendiente_promedio_pct",
        "temperatura_motor_c",
        "vibracion_mm_s",
        "presion_hidraulica_bar",
        "combustible_l_h",
        "horas_desde_mantenimiento",
        "alertas_previas_24h",
        "riesgo_parada_estimado",
        "parada_no_programada",
        "causa_probable",
    ]
    return data[columnas]


if __name__ == "__main__":
    salida = Path(__file__).resolve().parents[1] / "data" / "paradas_maquina_sintetico.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df = generar_data()
    df.to_csv(salida, index=False)
    tasa = df["parada_no_programada"].mean()
    print(f"Archivo generado: {salida}")
    print(f"Registros: {len(df):,}")
    print(f"Tasa de paradas no programadas: {tasa:.2%}")
