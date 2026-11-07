from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_image_generator 

ROOTH = Path(__file__).resolve().parents[1]
FIGURES_PATH = ROOTH / "reports" / "figures"

def plot_sample_images(generator, num_samples=16, seed=42):
    """
    Visualiza una muiestra aleatoria de imágenes junto con sus edades correspondientes.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(generator.samples, size=min(num_samples,generator.samples)
                         , replace=False)

    filenames = np.asarray(generator.filepaths)[indices]
    ages = np.asarray(generator.labels)[indices]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    from PIL import Image
    for ax, filename, age in zip(axes.flat, filenames, ages):
        with Image.open(filename) as img:
            img = img.convert("RGB")  # Asegurarse de que la imagen esté en formato RGB
            img = img.resize((224, 224))  # Redimensionar la imagen a 224x224
            ax.imshow(img)

        ax.set_title(f"Edad: {int(age)}años", fontsize=10)
        ax.axis('off')

        #ocultar cuadros no utilizados
    for ax in axes.flat[num_samples:]:
        ax.axis('off')

    fig.suptitle("Muestra aleatoria de fotografías", fontsize=16)
    fig.tight_layout()

    FIGURES_PATH.mkdir(parents=True, exist_ok=True) 
    fig.savefig(FIGURES_PATH / "sample_images.png", dpi=300, bbox_inches='tight')

    plt.show()

def main():
    """
    Función principal para cargar el generador de imágenes y visualizar una muestra.
    """
    generator = load_image_generator()
    print(f"Fotografías encontradas: {generator.samples}")
    plot_sample_images(generator,num_samples=16, seed=42)

if __name__ == "__main__":
    main()  