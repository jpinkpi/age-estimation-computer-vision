from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data_loader import load_data, validate_data

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

def main():
    """
    Función principal para cargar el generador de imágenes y mostrar información básica.
    """
    generator = load_image_generator()
    print(f"Registros encontrados en el csv:{len(load_data())}")
    print(f"Fotografías encontradas: {generator.samples}")

if __name__ == "__main__":
    main()
