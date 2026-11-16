from pathlib import Path
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,CSVLogger)
from src.data_loader import load_data, validate_data
from src.preprocessing import split_data, create_image_generators
from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "models"
REPORTS_PATH = ROOT / "reports" 

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 5

#CONFIGURACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD

def set_random_seed(seed=RANDOM_STATE):
    """
    Configura la semilla aleatoria para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#CONFIGURACION DE CALLBACKS 
def create_callbacks():
    """
    Configura los callbacks del entrenamiento.

    EarlyStopping:
        Detiene el entrenamiento si no mejora el MAE.

    ModelCheckpoint:
        Guarda el modelo con mejor MAE de validación.

    CSVLogger:
        Registra las métricas de cada época.
    """

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=PATIENCE,
            mode="min",
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=str(MODELS_PATH / "best_model.keras"),
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        CSVLogger(
            filename=str(REPORTS_PATH / "train_history.csv")
        )
    ]

    return callbacks

#preparacion de entrenamiento 
def prepare_training_data(batch_size=BATCH_SIZE):
    """
    Carga y prepara los datos para el entrenamiento.
    """
    df = load_data()
    validate_data(df)
    train_df, val_df, test_df = split_data(df,test_size=0.15, val_size=0.15, seed=RANDOM_STATE)
    train_gen, val_gen, test_gen = create_image_generators(train_df, val_df, test_df, batch_size=batch_size)

    #verificamos la correspondencia entre registro e imágenes
    for data, generator in[(train_df, train_gen), (val_df, val_gen), (test_df, test_gen)]:
        if len(data) != generator.samples:
            raise ValueError(f"El número de registros en el DataFrame ({len(data)}) no coincide con el número de imágenes en el generador ({generator.samples}).")

    return(train_gen, val_gen, test_gen, train_df, val_df, test_df)

#Entrnamiento del modelo
def train_model(model, train_gen, val_gen):
    """
    Entrena el modelo utilizando los generadores de datos.
    """
    callbacks = create_callbacks()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ============================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Ejecutar entrenamiento completo"
    )

    args = parser.parse_args()

    set_random_seed()

    print("\nDISPOSITIVOS DISPONIBLES")
    print(tf.config.list_physical_devices())

    print("\nPREPARACIÓN DE DATOS")

    (
        train_generator,
        val_generator,
        test_generator,
        train_df,
        val_df,
        test_df
    ) = prepare_training_data()

    print(f"Entrenamiento: {len(train_df)} imágenes")
    print(f"Validación: {len(val_df)} imágenes")
    print(f"Prueba: {len(test_df)} imágenes")

    print("\nCONSTRUCCIÓN DEL MODELO")

    model = build_model()
    model.summary()

    # No entrenar accidentalmente en CPU local
    if not args.train:
        print(
            "\nPreparación completada. "
            "No se ha iniciado el entrenamiento."
        )
        return

    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "No se detectó GPU. "
            "Ejecuta el entrenamiento en la plataforma GPU."
        )

    print("\nINICIANDO ENTRENAMIENTO")

    history = train_model(
        model,
        train_generator,
        val_generator
    )

    print("\nENTRENAMIENTO FINALIZADO")

    print(
        f"Mejor MAE de validación: "
        f"{min(history.history['val_mae']):.4f}"
    )

    print(
        f"Modelo guardado en: "
        f"{MODELS_PATH / 'best_model.keras'}"
    )

def verify_batch(generator):
    """
    Comprueba la estructura de un lote de entrenamiento.
    """

    images, ages = generator[0]

    print("\nVERIFICACIÓN DEL LOTE")

    print(f"Imágenes: {images.shape}")
    print(f"Edades: {ages.shape}")

    print(f"Tipo de imágenes: {images.dtype}")
    print(f"Tipo de edades: {ages.dtype}")

    if images.shape[1:] != (224, 224, 3):
        raise ValueError(
            "Las dimensiones no coinciden con ResNet50."
        )

    if len(images) != len(ages):
        raise ValueError(
            "El número de imágenes y edades no coincide."
        )

    if not np.isfinite(images).all():
        raise ValueError(
            "Se encontraron valores no finitos en las imágenes."
        )

    if not np.isfinite(ages).all():
        raise ValueError(
            "Se encontraron edades no finitas."
        )

    print("Lote verificado correctamente.")

def smoke_test():
    """
    Ejecuta una prueba mínima de entrenamiento en CPU.

    Utiliza:
    - 16 fotografías para entrenamiento.
    - 16 fotografías para validación.
    - Una época.
    - Sin guardar modelos.

    No modifica el conjunto de prueba.
    """

    print("\nINICIANDO SMOKE TEST")

    # Cargar y validar datos
    df = load_data()
    validate_data(df)

    # Mantener la partición original
    train_df, val_df, test_df = split_data(
        df,
        test_size=0.15,
        val_size=0.15,
        seed=RANDOM_STATE
    )

    # Seleccionar muestras pequeñas y reproducibles
    train_sample = train_df.sample(
        n=16,
        random_state=RANDOM_STATE
    )

    val_sample = val_df.sample(
        n=16,
        random_state=RANDOM_STATE
    )

    # Crear generadores independientes
    train_gen, val_gen, _ = create_image_generators(
        train_sample,
        val_sample,
        test_df.iloc[:1],
        batch_size=16
    )

    print(f"Imágenes de entrenamiento: {train_gen.samples}")
    print(f"Imágenes de validación: {val_gen.samples}")

    # Construir el modelo
    model = build_model()

    # Ejecutar una época de entrenamiento
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        steps_per_epoch=1,
        validation_steps=1,
        verbose=1
    )

    print("\nSMOKE TEST FINALIZADO")

    print(
        f"MAE de entrenamiento: "
        f"{history.history['mae'][-1]:.4f}"
    )

    print(
        f"MAE de validación: "
        f"{history.history['val_mae'][-1]:.4f}"
    )

def main():
    """
    Coordina el pipeline de entrenamiento de ResNet50.

    1. Configura la reproducibilidad.
    2. Carga y prepara los datos.
    3. Verifica los generadores.
    4. Construye la arquitectura neuronal.
    5. Entrena el modelo si se solicita y hay GPU.
    6. Presenta los resultados del entrenamiento.
    """

    # 1. Configuración de argumentos
    parser = argparse.ArgumentParser(
        description="Entrenamiento de ResNet50 para estimación de edad."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Ejecutar entrenamiento completo con GPU."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Ejecutar una prueba mínima de entrenamiento en CPU."
    )

    args = parser.parse_args()

    if args.train and args.smoke_test:
        parser.error(
            "No puedes utilizar --train y --smoke-test simultáneamente."
        )
    # 2. Configuración de reproducibilidad
    set_random_seed()
    if args.smoke_test:
        smoke_test()
        return
    print("\nDISPOSITIVOS DISPONIBLES")
    print(tf.config.list_physical_devices())

    # 3. Preparación de datos
    print("\nPREPARACIÓN DE DATOS")

    (
        train_gen,
        val_gen,
        test_gen,
        train_df,
        val_df,
        test_df
    ) = prepare_training_data()

    print(f"Entrenamiento: {len(train_df)} imágenes")
    print(f"Validación: {len(val_df)} imágenes")
    print(f"Prueba: {len(test_df)} imágenes")

    # 4. Verificación de un lote de imágenes
    verify_batch(train_gen)

    # 5. Construcción del modelo
    print("\nCONSTRUCCIÓN DEL MODELO")

    model = build_model()
    model.summary()

    # 6. Ejecución sin entrenamiento
    if not args.train:
        print(
            "\nPipeline verificado correctamente."
            "\nNo se ha iniciado el entrenamiento."
        )
        return

    # 7. Comprobación de GPU
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "No se detectó una GPU disponible. "
            "Ejecuta el entrenamiento en la plataforma GPU."
        )

    # 8. Entrenamiento
    print("\nINICIANDO ENTRENAMIENTO")

    history = train_model(
        model,
        train_gen,
        val_gen
    )

    # 9. Resultados
    best_epoch = (
        np.argmin(history.history["val_mae"]) + 1
    )

    best_mae = min(history.history["val_mae"])

    print("\nENTRENAMIENTO FINALIZADO")

    print(f"Mejor época: {best_epoch}")
    print(f"Mejor MAE de validación: {best_mae:.4f} años")

    print(
        f"Modelo guardado en: "
        f"{MODELS_PATH / 'best_model.keras'}"
    )

    print(
        f"Historial guardado en: "
        f"{REPORTS_PATH / 'train_history.csv'}"
    )



if __name__ == "__main__":
    main()
