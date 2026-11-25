"""Run the pretrained AIZOO face-mask detector on a local webcam."""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
sys.path.insert(0, str(Path(__file__).parent / "vendor" / "aizoo"))
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "vendor" / "aizoo" / "models"
PROTO = MODEL_DIR / "face_mask_detection.prototxt"
WEIGHTS = MODEL_DIR / "face_mask_detection.caffemodel"
FEATURE_MAPS = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
ANCHOR_SIZES = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
ANCHOR_RATIOS = [[1, 0.62, 0.42]] * 5


def get_detections(net, frame, threshold=0.55, mask_bias=0.0):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rgb, scalefactor=1 / 255.0, size=(260, 260))
    net.setInput(blob)
    names = [net.getLayerNames()[int(index) - 1] for index in net.getUnconnectedOutLayers().flatten()]
    loc, classes = net.forward(names)
    anchors = np.expand_dims(generate_anchors(FEATURE_MAPS, ANCHOR_SIZES, ANCHOR_RATIOS), axis=0)
    boxes = decode_bbox(anchors, loc)[0]
    class_scores = classes[0]
    scores = np.max(class_scores, axis=1)
    adjusted_scores = class_scores.copy()
    adjusted_scores[:, 0] += mask_bias
    class_ids = np.argmax(adjusted_scores, axis=1)
    valid = np.flatnonzero(scores > threshold)
    if not valid.size:
        return []
    h, w = frame.shape[:2]
    results = []
    keep = single_class_non_max_suppression(boxes[valid], scores[valid], conf_thresh=threshold, iou_thresh=0.4)
    for index in keep:
        source_i = valid[index]
        x1, y1, x2, y2 = boxes[source_i]
        coords = (max(0, int(x1 * w)), max(0, int(y1 * h)), min(w, int(x2 * w)), min(h, int(y2 * h)))
        results.append((coords, int(class_ids[source_i]), float(class_scores[source_i, class_ids[source_i]])))
    return results


class MaskDetector:
    def __init__(self):
        if not PROTO.exists() or not WEIGHTS.exists():
            raise FileNotFoundError("AIZOO model files are missing. Check vendor/aizoo/models/.")
        self.net = cv2.dnn.readNet(str(WEIGHTS), str(PROTO))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def process(self, frame, threshold, mask_bias):
        detections = get_detections(self.net, frame, threshold, mask_bias)
        annotated = frame.copy()
        for (x1, y1, x2, y2), class_id, confidence in detections:
            label = "Mask" if class_id == 0 else "No Mask"
            color = (50, 205, 115) if class_id == 0 else (65, 95, 245)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label}  {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            top = max(y1, text_h + 12)
            cv2.rectangle(annotated, (x1, top - text_h - 10), (x1 + text_w + 12, top), color, -1)
            cv2.putText(annotated, text, (x1 + 6, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return annotated, detections


class MaskDetectorApp:
    BG = "#0b1220"
    PANEL = "#121d30"
    PANEL_2 = "#18263b"
    TEXT = "#edf4ff"
    MUTED = "#9cafc7"
    TEAL = "#36d6bd"

    def __init__(self, root):
        self.root = root
        self.root.title("MaskWatch | Offline Face Mask Detection")
        self.root.geometry("1120x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.BG)
        self.detector = MaskDetector()
        self.camera = None
        self.camera_running = False
        # Recall-oriented defaults: admit weaker detections and mildly favor Mask.
        self.threshold = tk.IntVar(value=40)
        self.mask_sensitivity = tk.IntVar(value=20)
        self.preview_photo = None
        self.annotated_frame = None
        self.source_frame = None
        self.source_name = ""
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.set_status("Ready · image processing stays on this laptop", self.TEAL)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(22, 18))
        logo = tk.Canvas(header, width=42, height=42, bg=self.BG, highlightthickness=0)
        logo.create_oval(3, 3, 39, 39, fill=self.PANEL_2, outline=self.TEAL, width=2)
        logo.create_text(21, 21, text="M", fill=self.TEAL, font=("Segoe UI", 17, "bold"))
        logo.pack(side="left", padx=(0, 12))
        titlebox = tk.Frame(header, bg=self.BG)
        titlebox.pack(side="left")
        tk.Label(titlebox, text="MaskWatch", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(titlebox, text="Offline face mask detection", bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(header, text="  LOCAL AI  ·  CPU  ", bg=self.PANEL_2, fg=self.TEAL,
                 font=("Segoe UI", 9, "bold"), padx=12, pady=8).pack(side="right", pady=5)

        toolbar = tk.Frame(self.root, bg=self.BG)
        toolbar.pack(fill="x", padx=28, pady=(0, 12))
        self._button(toolbar, "▣  Import Image", self.import_image, self.PANEL_2, self.TEXT).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        self._button(toolbar, "▶  Start Webcam", self.start_camera, self.TEAL, self.BG).pack(
            side="left", fill="x", expand=True, padx=6)
        self._button(toolbar, "■  Stop Webcam", self.stop_camera, self.PANEL_2, self.TEXT).pack(
            side="left", fill="x", expand=True, padx=6)
        self._button(toolbar, "↓  Save Result", self.save_result, self.PANEL_2, self.TEXT).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=28, pady=(0, 18))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        preview_card = tk.Frame(body, bg=self.PANEL, padx=16, pady=15)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_card.grid_rowconfigure(1, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)
        tk.Label(preview_card, text="DETECTION PREVIEW", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        self.preview = tk.Label(preview_card, text="No source selected\n\nClick Import Image or Start Webcam above",
                                bg="#070d17", fg="#71839b", font=("Segoe UI", 13),
                                justify="center", compound="center")
        self.preview.grid(row=1, column=0, sticky="nsew")

        side = tk.Frame(body, bg=self.BG, width=280)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        results = tk.Frame(side, bg=self.PANEL, padx=13, pady=11)
        results.pack(fill="x", pady=(0, 12))
        tk.Label(results, text="RESULTS", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.result_status = tk.Label(results, text="Waiting for an image", bg=self.PANEL, fg=self.TEXT,
                                      font=("Segoe UI", 14, "bold"), wraplength=240, justify="left")
        self.result_status.pack(anchor="w", pady=(6, 8))
        self.mask_count = self._count_box(results, "MASK", "0", "#4edc9b")
        self.no_mask_count = self._count_box(results, "NO MASK", "0", "#ff727c")
        self.face_count = tk.Label(results, text="0 faces detected", bg=self.PANEL, fg=self.MUTED,
                                   font=("Segoe UI", 9))
        self.face_count.pack(anchor="w", pady=(7, 0))

        settings = tk.Frame(side, bg=self.PANEL, padx=13, pady=11)
        settings.pack(fill="x", pady=(0, 12))
        tk.Label(settings, text="CONFIDENCE FILTER", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.threshold_text = tk.Label(settings, text="40%", bg=self.PANEL, fg=self.TEXT,
                                       font=("Segoe UI", 16, "bold"))
        self.threshold_text.pack(anchor="w", pady=(8, 2))
        tk.Scale(settings, from_=30, to=90, orient="horizontal", resolution=5, variable=self.threshold,
                 command=self._threshold_changed, bg=self.PANEL, fg=self.MUTED, troughcolor=self.PANEL_2,
                 activebackground=self.TEAL, highlightthickness=0, bd=0, showvalue=False,
                 sliderlength=18).pack(fill="x")
        tk.Label(settings, text="Higher values show fewer, stronger detections.", bg=self.PANEL,
                 fg=self.MUTED, font=("Segoe UI", 8), wraplength=240, justify="left").pack(anchor="w")
        tk.Label(settings, text="MASK SENSITIVITY", bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(13, 0))
        self.sensitivity_text = tk.Label(settings, text="20%", bg=self.PANEL, fg=self.TEXT,
                                         font=("Segoe UI", 14, "bold"))
        self.sensitivity_text.pack(anchor="w", pady=(5, 0))
        tk.Scale(settings, from_=0, to=40, orient="horizontal", resolution=5, variable=self.mask_sensitivity,
                 command=self._sensitivity_changed, bg=self.PANEL, fg=self.MUTED, troughcolor=self.PANEL_2,
                 activebackground=self.TEAL, highlightthickness=0, bd=0, showvalue=False,
                 sliderlength=18).pack(fill="x")
        tk.Label(settings, text="Raise this to favor Mask labels. Higher sensitivity can add false Mask results.",
                 bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8), wraplength=240,
                 justify="left").pack(anchor="w")

        footer = tk.Frame(self.root, bg=self.BG)
        footer.pack(fill="x", padx=30, pady=(0, 14))
        self.status_label = tk.Label(footer, text="", bg=self.BG, fg=self.MUTED,
                                     font=("Segoe UI", 9), anchor="w")
        self.status_label.pack(fill="x")

    def _count_box(self, parent, title, value, color):
        box = tk.Frame(parent, bg=self.PANEL_2, padx=12, pady=9)
        box.pack(fill="x", pady=4)
        tk.Label(box, text=title, bg=self.PANEL_2, fg=color,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        number = tk.Label(box, text=value, bg=self.PANEL_2, fg=self.TEXT,
                          font=("Segoe UI", 14, "bold"))
        number.pack(side="right")
        return number

    def _button(self, parent, text, command, bg, fg):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=bg, activeforeground=fg, relief="flat", bd=0,
                         cursor="hand2", font=("Segoe UI", 10, "bold"), pady=10)

    def _threshold_changed(self, value):
        self.threshold_text.configure(text=f"{int(float(value))}%")
        if not self.camera_running and self.source_frame is not None:
            self.run_detection(self.source_frame, self.source_name)

    def _sensitivity_changed(self, value):
        self.sensitivity_text.configure(text=f"{int(float(value))}%")
        if not self.camera_running and self.source_frame is not None:
            self.run_detection(self.source_frame, self.source_name)

    def set_status(self, text, color=None):
        self.status_label.configure(text=text, fg=color or self.MUTED)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(300, self.preview.winfo_width() - 16)
        max_h = max(220, self.preview.winfo_height() - 16)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def update_results(self, detections):
        masks = sum(1 for _, class_id, _ in detections if class_id == 0)
        no_masks = len(detections) - masks
        self.mask_count.configure(text=str(masks))
        self.no_mask_count.configure(text=str(no_masks))
        self.face_count.configure(text=f"{len(detections)} face{'s' if len(detections) != 1 else ''} detected")
        if not detections:
            self.result_status.configure(text="No face detected", fg=self.TEXT)
        elif no_masks and masks:
            self.result_status.configure(text="Mixed results", fg="#ffd166")
        elif no_masks:
            self.result_status.configure(text="No mask detected", fg="#ff8b91")
        else:
            self.result_status.configure(text="Mask detected", fg="#4edc9b")

    def run_detection(self, frame, source):
        self.source_frame = frame.copy()
        self.source_name = source
        self.annotated_frame, detections = self.detector.process(
            frame, self.threshold.get() / 100, self.mask_sensitivity.get() / 100)
        self.update_results(detections)
        self.show_frame(self.annotated_frame)
        self.set_status(f"Showing {source} · {frame.shape[1]}×{frame.shape[0]} · processed locally")

    def import_image(self):
        path = filedialog.askopenfilename(title="Choose a photo", filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        self.stop_camera()
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("OpenCV could not read this image format.")
            self.run_detection(frame, Path(path).name)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)

    def start_camera(self):
        if self.camera_running:
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror("Camera unavailable", "Check that your webcam is connected and allowed in Windows camera settings.", parent=self.root)
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_running = True
        self.set_status("Webcam is active · processed locally", self.TEAL)
        self._camera_tick()

    def _camera_tick(self):
        if not self.camera_running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if ok:
            self.run_detection(frame, "webcam")
        else:
            self.stop_camera()
            self.set_status("Camera frame unavailable", "#ff8b91")
        if self.camera_running:
            self.root.after(20, self._camera_tick)

    def stop_camera(self):
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def save_result(self):
        if self.annotated_frame is None:
            messagebox.showinfo("Nothing to save", "Import an image or start the webcam first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(title="Save annotated result", defaultextension=".jpg",
                                             filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")])
        if not path:
            return
        extension = Path(path).suffix.lower() or ".jpg"
        ok, encoded = cv2.imencode(extension, self.annotated_frame)
        if ok:
            encoded.tofile(path)
            self.set_status(f"Saved annotated image · {Path(path).name}", self.TEAL)
        else:
            messagebox.showerror("Save failed", "Could not encode the result image.", parent=self.root)

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    if not PROTO.exists() or not WEIGHTS.exists():
        raise SystemExit("AIZOO model files are missing. Check vendor/aizoo/models/.")
    root = tk.Tk()
    try:
        MaskDetectorApp(root)
    except Exception as exc:
        root.destroy()
        raise SystemExit(f"Could not start MaskWatch: {exc}")
    root.mainloop()


if __name__ == "__main__":
    main()
