'''
Arquitectura de red neuronal para estimación de edad.
Modelo:
- ResNet50 preentrenada con ImageNet.
- Aprendizaje por transferencia.
- Cabezal de regresión para estimación de edad.
'''

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

IMAGE_SHAPE = (224, 224, 3)

LEARNING_RATE = 0.001


# ============================================================
# 2. CONSTRUCCIÓN DEL MODELO
# ============================================================

def build_model(
    input_shape=IMAGE_SHAPE,
    learning_rate=LEARNING_RATE,
    dropout_rate=0.3,
    train_base=False
):
    """
    Construye una red neuronal para estimar edades.

    Parameters
    ----------
    input_shape : tuple
        Dimensiones de las imágenes de entrada.

    learning_rate : float
        Tasa de aprendizaje del optimizador.

    dropout_rate : float
        Proporción de desactivación aleatoria de neuronas.

    train_base : bool
        Indica si las capas de ResNet50 serán entrenables.

    Returns
    -------
    keras.Model
        Modelo compilado.
    """

    # --------------------------------------------------------
    # 2.1 Modelo base
    # --------------------------------------------------------

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Congelar o habilitar entrenamiento de ResNet50
    base_model.trainable = train_base

    # --------------------------------------------------------
    # 2.2 Arquitectura de regresión
    # --------------------------------------------------------

    inputs = Input(shape=input_shape)

    # Mantener BatchNormalization en modo inferencia
    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(dropout_rate)(x)

    # Salida numérica para regresión
    outputs = Dense(
        1,
        activation="linear"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="age_estimation_resnet50"
    )

    # --------------------------------------------------------
    # 2.3 Compilación
    # --------------------------------------------------------

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"]
    )

    return model


# ============================================================
# 3. EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    model = build_model()

    print("\nARQUITECTURA DEL MODELO")
    model.summary()

    print("\nModelo construido correctamente.")


if __name__ == "__main__":
    main()