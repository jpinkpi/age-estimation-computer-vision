from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from src.data_loader import load_data, validate_data
from sklearn.model_selection import train_test_split
import pandas as pd



ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "photos" # Directorio que contiene las imágenes

def load_image_generator(batch_size=32):
    """
    Carga imágenes y edades ,mediante ImageDataGenerator"""
    df = load_data()
    validate_data(df)

    generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df,
        directory=DATA_PATH,
        x_col="file_name",
        y_col="real_age",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False,
        validate_filenames=True
    )
    return generator

from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.15, val_size=0.15, seed=42):
    """
    Divide el dataset en entrenamiento, validación y prueba.
    """

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed
    )

    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed
    )
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size debe estar entre 0 y 1.")

    if test_size + val_size >= 1:
        raise ValueError(
        "La suma de test_size y val_size debe ser menor que 1."
    )
    return train_df, val_df, test_df



def create_image_generators(
    train_df,
    val_df,
    test_df,
    batch_size=32
):
    """
    Crea generadores independientes para
    entrenamiento, validación y prueba.
    """

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    common_params = {
        "directory": str(DATA_PATH),
        "x_col": "file_name",
        "y_col": "real_age",
        "target_size": (224, 224),
        "batch_size": batch_size,
        "class_mode": "raw",
        "color_mode": "rgb",
        "validate_filenames": True
    }

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        seed=42,
        **common_params
    )

    val_generator = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        shuffle=False,
        **common_params
    )

    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        shuffle=False,
        **common_params
    )

    return train_generator, val_generator, test_generator

def verify_split(train_df, val_df, test_df):
    """
    Verifica la integridad y distribución de los conjuntos.

    Comprueba:
    - Tamaños y proporciones.
    - Ausencia de fotografías repetidas entre conjuntos.
    - Distribución de edades.
    - Representación de menores de edad.

    """
    datasets = {"Entrenamiento": train_df, "Validación": val_df, "Prueba": test_df}
    total_samples = sum(len(df) for df in datasets.values())

    #1 Tamaños y proporciones
    print("\nDistribución de conjuntos:")
    for name, df in datasets.items():
        percentage = (len(df) / total_samples) * 100
        print(f"{name}: {len(df)} observaciones"
              f"({percentage:.2f}% del total)")
               # 2. Verificar ausencia de archivos repetidos
    train_files = set(train_df["file_name"])
    val_files = set(val_df["file_name"])
    test_files = set(test_df["file_name"])

    if not train_files.isdisjoint(val_files):
        raise ValueError(
            "Hay fotografías compartidas entre entrenamiento y validación."
        )

    if not train_files.isdisjoint(test_files):
        raise ValueError(
            "Hay fotografías compartidas entre entrenamiento y prueba."
        )

    if not val_files.isdisjoint(test_files):
        raise ValueError(
            "Hay fotografías compartidas entre validación y prueba."
        )

    print("\nNo existen fotografías compartidas entre conjuntos.")

    # 3. Estadísticas descriptivas
    print("\nEstadísticas de edad por conjunto:")

    statistics = pd.DataFrame({
        name: df["real_age"].describe()
        for name, df in datasets.items()
    })

    print(statistics.round(2))

    # 4. Proporción de menores
    print("\nRepresentación de menores de edad:")

    for name, df in datasets.items():
        minors_percentage = (df["real_age"] < 18).mean() * 100

        print(
            f"{name}: {minors_percentage:.2f}% menores de edad"
        )

    return statistics

def main():

    # Cargar y validar el dataset
    df = load_data()
    validate_data(df)

    # Dividir los datos (70/15/15)
    train_df, val_df, test_df = split_data(df)

    # Verificar las particiones
    verify_split(
        train_df,
        val_df,
        test_df
    )

    # Crear los generadores de imágenes
    train_gen, val_gen, test_gen = create_image_generators(
        train_df,
        val_df,
        test_df
    )

    # Verificar la cantidad de fotografías
    print("\nGeneradores de imágenes:")

    print(f"Entrenamiento: {train_gen.samples}")
    print(f"Validación: {val_gen.samples}")
    print(f"Prueba: {test_gen.samples}")

    # Comprobar correspondencia entre etiquetas e imágenes
    assert train_gen.samples == len(train_df)
    assert val_gen.samples == len(val_df)
    assert test_gen.samples == len(test_df)

    print("\nPreprocesamiento completado correctamente.")


if __name__ == "__main__":
    main()