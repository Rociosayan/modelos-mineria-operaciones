import sqlite3
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DB_URL = (
    "https://raw.githubusercontent.com/Rociosayan/"
    "PMD1_Fundamentos_Gestion_Datos/main/casos/"
    "16_abarrotes_ventas_inventario/abarrotes_ventas_inventario.db"
)
DB_PATH = Path(__file__).with_name("abarrotes_ventas_inventario.db")


def descargar_base_si_no_existe() -> None:
    if not DB_PATH.exists():
        urllib.request.urlretrieve(DB_URL, DB_PATH)


def limpiar_numero(serie: pd.Series) -> pd.Series:
    """Convierte valores como '8 aprox' o '15.06' a numerico."""
    extraido = serie.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(extraido, errors="coerce")


def limpiar_texto(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": np.nan, "none": np.nan})
    )


def parsear_fecha_mixta(serie: pd.Series) -> pd.Series:
    texto = serie.astype(str).str.strip()
    resultado = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    es_iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    resultado.loc[es_iso] = pd.to_datetime(
        texto.loc[es_iso], format="%Y-%m-%d", errors="coerce"
    )
    resultado.loc[~es_iso] = pd.to_datetime(
        texto.loc[~es_iso], errors="coerce", dayfirst=True
    )
    return resultado


def cargar_y_limpiar_datos(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM ventas_original", conn)

    columnas_numericas = [
        "cantidad_unidades",
        "precio_venta_unitario",
        "costo_unitario",
        "descuento_pct",
        "stock_inicial",
        "stock_final",
        "merma_unidades",
        "monto_venta_soles",
        "margen_venta_soles",
        "productos_costo_unitario",
        "productos_precio_lista",
    ]
    for columna in columnas_numericas:
        df[columna] = limpiar_numero(df[columna])

    columnas_categoricas = [
        "canal",
        "metodo_pago",
        "tiendas_zona",
        "tiendas_tipo_local",
        "productos_categoria",
        "clientes_segmento",
    ]
    for columna in columnas_categoricas:
        df[columna] = limpiar_texto(df[columna])

    df["fecha_operacion_dt"] = parsear_fecha_mixta(df["fecha_operacion"])
    df["mes_operacion"] = df["fecha_operacion_dt"].dt.month
    df["dia_semana"] = df["fecha_operacion_dt"].dt.dayofweek
    df["fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["importe_bruto_soles"] = df["cantidad_unidades"] * df["precio_venta_unitario"]
    df["importe_con_descuento_soles"] = df["importe_bruto_soles"] * (
        1 - df["descuento_pct"].fillna(0) / 100
    )
    df["utilidad_unitaria_soles"] = df["precio_venta_unitario"] - df["costo_unitario"]

    df = df.dropna(subset=["monto_venta_soles"]).copy()
    return df


def crear_pipeline(columnas_numericas: list[str], columnas_categoricas: list[str]) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columnas_numericas,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                columnas_categoricas,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", LinearRegression()),
        ]
    )


def obtener_coeficientes(modelo: Pipeline, columnas_numericas: list[str]) -> pd.DataFrame:
    preprocesador = modelo.named_steps["preprocesamiento"]
    regresion = modelo.named_steps["modelo"]
    nombres = preprocesador.get_feature_names_out()
    nombres = [n.replace("num__", "").replace("cat__", "") for n in nombres]

    return (
        pd.DataFrame({"variable": nombres, "coeficiente": regresion.coef_})
        .assign(abs_coeficiente=lambda d: d["coeficiente"].abs())
        .sort_values("abs_coeficiente", ascending=False)
        .drop(columns="abs_coeficiente")
    )


def main() -> None:
    descargar_base_si_no_existe()

    conn = sqlite3.connect(DB_PATH)
    tablas = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    print("Tablas disponibles en SQLite:")
    print(tablas.to_string(index=False))

    df = cargar_y_limpiar_datos(conn)
    print(f"\nDatos cargados desde ventas_original: {df.shape[0]} filas x {df.shape[1]} columnas")

    columnas_numericas = [
        "cantidad_unidades",
        "precio_venta_unitario",
        "costo_unitario",
        "descuento_pct",
        "stock_inicial",
        "stock_final",
        "merma_unidades",
        "productos_costo_unitario",
        "productos_precio_lista",
        "importe_bruto_soles",
        "importe_con_descuento_soles",
        "utilidad_unitaria_soles",
        "mes_operacion",
        "dia_semana",
        "fin_semana",
    ]
    columnas_categoricas = [
        "canal",
        "metodo_pago",
        "tiendas_zona",
        "tiendas_tipo_local",
        "productos_categoria",
        "clientes_segmento",
    ]
    target = "monto_venta_soles"

    X = df[columnas_numericas + columnas_categoricas]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    modelo = crear_pipeline(columnas_numericas, columnas_categoricas)
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = float(np.sqrt(mse_test))

    print("\nMetricas del modelo")
    print(f"R2 train : {r2_train:.4f}")
    print(f"R2 test  : {r2_test:.4f}")
    print(f"MSE test : {mse_test:.2f}")
    print(f"RMSE test: S/ {rmse_test:.2f}")

    coeficientes = obtener_coeficientes(modelo, columnas_numericas)
    print("\nCoeficientes mas influyentes:")
    print(coeficientes.head(12).to_string(index=False))

    resultados = X_test.copy()
    resultados.insert(0, "id_venta", df.loc[X_test.index, "id_venta"].values)
    resultados["monto_real_soles"] = y_test.values
    resultados["monto_predicho_soles"] = np.round(pred_test, 2)
    resultados["residuo_soles"] = np.round(y_test.values - pred_test, 2)
    resultados["error_abs_soles"] = resultados["residuo_soles"].abs()

    df_limpio = df[
        ["id_venta", "fecha_operacion", target]
        + columnas_numericas
        + columnas_categoricas
    ].copy()
    df_limpio.to_sql("ventas_limpias_regresion", conn, if_exists="replace", index=False)
    resultados.to_sql("predicciones_regresion_lineal", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred_test, alpha=0.75, edgecolor="white")
    minimo = min(y_test.min(), pred_test.min())
    maximo = max(y_test.max(), pred_test.max())
    plt.plot([minimo, maximo], [minimo, maximo], "r--", label="Prediccion perfecta")
    plt.title(f"Regresion lineal: ventas reales vs predichas\nR2 test = {r2_test:.3f}")
    plt.xlabel("Monto real de venta (S/)")
    plt.ylabel("Monto predicho de venta (S/)")
    plt.legend()
    plt.tight_layout()
    grafico_path = DB_PATH.with_name("regresion_lineal_abarrotes.png")
    plt.savefig(grafico_path, dpi=140)

    print("\nTablas guardadas en SQLite:")
    print("- ventas_limpias_regresion")
    print("- predicciones_regresion_lineal")
    print(f"Grafico guardado en: {grafico_path}")


if __name__ == "__main__":
    main()
