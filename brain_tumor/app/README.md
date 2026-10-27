# App web: clasificador de tumores cerebrales

Subes una resonancia magnética cerebral y la app devuelve la clase
(glioma, meningioma, sin tumor, hipófisis) con sus probabilidades.
Usa el modelo `brain_tumor_resnet50.pth` generado por `../train_resnet50.py`.

> Proyecto académico. No es una herramienta de diagnóstico médico.

## Opción A: Google Colab (rápida, enlace temporal)

1. Sube `app.py` y `brain_tumor_resnet50.pth` a la misma carpeta del entorno
   (o ejecuta la app en el mismo Colab donde entrenaste).
2. En una celda:
   ```
   !pip install -q gradio
   !python app.py
   ```
3. Abre el enlace `https://....gradio.live` que aparece. Dura unas 72 h
   y deja de funcionar al desconectar el entorno.

## Opción B: Hugging Face Spaces (gratis y permanente)

1. Crea una cuenta en https://huggingface.co y un *Space* nuevo con SDK **Gradio**
   (hardware gratuito *CPU basic*).
2. Sube al Space: `app.py`, `requirements.txt` y `brain_tumor_resnet50.pth`
   (~95 MB; la web de Hugging Face lo acepta directamente).
3. El Space se construye solo y queda disponible en
   `https://huggingface.co/spaces/<usuario>/<nombre>`.

## Opción C: en tu ordenador

```
pip install -r requirements.txt
python app.py
```

y abre http://127.0.0.1:7860. Con `MODEL_PATH=/ruta/al/modelo.pth` puedes
indicar otra ubicación del modelo.
