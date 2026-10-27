# -*- coding: utf-8 -*-
"""
App web para clasificar resonancias magnéticas cerebrales en 4 clases
(glioma, meningioma, notumor, pituitary) con la ResNet-50 entrenada en
train_resnet50.py.

Uso:
  - Local / Hugging Face Spaces:  python app.py
  - Google Colab:                 !python app.py   (genera un enlace público)

El modelo se lee de MODEL_PATH (por defecto 'brain_tumor_resnet50.pth' junto a
este archivo).
"""
import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(HERE, 'brain_tumor_resnet50.pth'))

# Mismo orden que ImageFolder (alfabético) durante el entrenamiento
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NOMBRES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'Sin tumor',
    'pituitary': 'Tumor de hipófisis (pituitario)',
}
# Por debajo de esta probabilidad se avisa de que la predicción es poco fiable
UMBRAL_CONFIANZA = 0.70

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mismo preprocesado que transform_eval en el entrenamiento
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encuentra el modelo en '{MODEL_PATH}'. Copia aquí "
        "'brain_tumor_resnet50.pth' o indica la ruta con la variable MODEL_PATH.")
model = load_model(MODEL_PATH)


def predict(image):
    if image is None:
        return {}, "Sube una imagen de resonancia magnética."

    x = preprocess(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        # Test-Time Augmentation: original + volteo horizontal
        probs = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    probs = probs[0].cpu()

    resultado = {NOMBRES[c]: float(p) for c, p in zip(CLASSES, probs)}
    top = int(probs.argmax())
    conf = float(probs[top])

    mensaje = f"**Predicción:** {NOMBRES[CLASSES[top]]} ({conf*100:.1f}%)"
    if conf < UMBRAL_CONFIANZA:
        mensaje += ("\n\n⚠️ Confianza baja: el modelo no está seguro. Puede que la "
                    "imagen no sea una resonancia cerebral axial o que sea un caso difícil.")
    return resultado, mensaje


DESCRIPCION = """
Sube una resonancia magnética cerebral (corte 2D) y el modelo indicará si
corresponde a **glioma**, **meningioma**, **tumor de hipófisis** o **sin tumor**.

> ⚠️ Proyecto académico. **No es una herramienta de diagnóstico médico.**
> El modelo solo conoce estas 4 clases: ante cualquier otra imagen responderá
> igualmente una de ellas.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label="Resonancia magnética"),
    outputs=[
        gr.Label(num_top_classes=4, label="Probabilidades"),
        gr.Markdown(),
    ],
    title="Clasificador de tumores cerebrales (ResNet-50)",
    description=DESCRIPCION,
    submit_btn="Clasificar",
    clear_btn="Limpiar",
    flagging_mode='never',
)

if __name__ == '__main__':
    in_colab = 'COLAB_RELEASE_TAG' in os.environ
    demo.launch(share=in_colab)
