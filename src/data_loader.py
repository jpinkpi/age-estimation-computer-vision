from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] 
DATA_PATH = ROOT / "data" / "raw" / "labels.csv" 

def load_data(path=DATA_PATH):
    """
    Carga los datos del archivo CSV y devuelve un DataFrame de pandas.
    parameters:
    ----------
    path : str or path 
        La ruta al archivo CSV que contiene los datos.

    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo en la ruta especificada: {path}")
    df= pd.read_csv(path)
    return df

def validate_data(df):
    """
    Valida el DataFrame para asegurarse de que contiene las columnas esperadas.
    parameters:
    ----------
    df : pandas.DataFrame
        El DataFrame a validar.
    comprueba:
    - que el DataFrame contiene las columnas 'file_name' y 'real_age'.
    - valores faltantes
    - Nombres de las fotografías duplicados
    - Edades fuera del intervalo esperado

    """
    required_columns = {"file_name", "real_age"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing_columns}")
    if df.empty:
        raise ValueError("El DataFrame está vacío.")
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores faltantes.")
    if df.duplicated(subset=["file_name"]).any():
        raise ValueError("El DataFrame contiene nombres de fotografías duplicados.")
    if not df["real_age"].between(0, 120).all():
        raise ValueError("El DataFrame contiene edades fuera del intervalo esperado (0-120).")

    return True

def get_data_info(df):
    """
    Devuelve información básica sobre el DataFrame.
    parameters:
    ----------
    df : pandas.DataFrame
        El DataFrame del que se desea obtener información.

    returns:
    -------
    dict
        Un diccionario con la información básica del DataFrame, incluyendo el número de filas, columnas, tipos de datos y estadísticas descriptivas.
    """
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "min_age": df["real_age"].min(),
        "max_age": df["real_age"].max(),
        "mean_age": df["real_age"].mean(),
        "median_age": df["real_age"].median(),
        "std_age": df["real_age"].std()}
    return info

def main():
    df = load_data()
    validate_data(df)
    info = get_data_info(df)
    print("Información del DataFrame:")

    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()