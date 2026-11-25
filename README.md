# Offline Face Mask Detection

This project uses AIZOO's small, pretrained deep-learning model to find faces and label each as **Mask** or **No Mask**. The desktop interface supports live webcam detection and imported photos, with result counts, an adjustable confidence filter, and saving annotated images. You do not need to collect images or train a model. The laptop in the supplied photo can run the model on its CPU.

The useful pretrained model and inference helper files have been cloned into `vendor/aizoo/` from the [AIZOOTech FaceMaskDetection project](https://github.com/AIZOOTech/FaceMaskDetection). Once the Python packages are installed, the webcam detector runs offline; it does not need a first-run model download. The published project describes its model as 1.01 million parameters and reports training on 7,971 images. It is provided under the MIT license; see `THIRD_PARTY_NOTICES.md`.

## Setup

Python 3.10 or 3.11 (64-bit) is recommended. In PowerShell, from this project folder, enter these commands one at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Installing packages needs internet once. No internet is needed when running the detector afterward.

## Run

```powershell
python detect.py
```

Use **Start Webcam** for live detection or **Import Image** to analyze a saved photo. The results panel shows the face count and mask/no-mask counts. **Confidence Filter** controls which detections are shown; lower settings catch weaker detections but can add false alarms. **Mask Sensitivity** gives the Mask class an adjustable preference; raise it if masked faces are being called No Mask. A higher setting can also create false Mask results, so compare both labels on known examples before choosing a setting. The defaults favor catching more possible masks. Use **Save Result Image** to export the annotated frame. The camera image is processed locally and is not uploaded.

## Try it

Test with several people facing the webcam, both with and without masks. Use good lighting and keep faces reasonably close to the camera. Turned-away, small, or poorly lit faces can be missed, and classifications can be wrong. The sensitivity control changes the class decision; it does not retrain the model or guarantee higher accuracy.

This is a student demonstration, not a health or security system. For your group presentation, explain that you are using an existing pretrained CNN, so the project focuses on local inference and evaluation rather than training a new model.

## Included AIZOO files

- `vendor/aizoo/models/face_mask_detection.caffemodel` and `face_mask_detection.prototxt` — pretrained network.
- `vendor/aizoo/utils/` — anchor decoding and non-maximum suppression used to turn network outputs into boxes.
- `vendor/aizoo/LICENSE` — upstream license.
