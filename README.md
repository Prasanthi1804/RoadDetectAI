<div align="center">

# 🛣️ Road DetectAI

### AI-Powered Pothole Detection for Smarter Road Inspection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

Road DetectAI detects potholes in road images, draws bounding boxes with confidence scores, estimates damage severity, and lets you download annotated images and CSV reports.

</div>

---

## ✨ Features

- 🎯 **Pothole detection** with a YOLOv8n model
- 📦 **Bounding boxes and confidence scores** for every detection
- 📐 **Severity estimation** based on how much of the image is covered
- 📊 **Interactive Streamlit dashboard** with metrics and a results table
- ⬇️ **Downloads** for CSV reports and annotated images
- 🎚️ **Adjustable confidence threshold** from the sidebar

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **YOLOv8 (Ultralytics)** | Object detection |
| **Streamlit** | Web dashboard |
| **Pandas** | Results table and CSV export |
| **Pillow** | Image handling |
| **OpenCV** | Image processing |

---

## 📁 Project Structure

```text
Pothole-and-Road-defect-Detector/
├── app.py                  # Streamlit dashboard
├── severity.py             # Severity calculation
├── train.py                # YOLOv8 training script
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (libgl1, libglib2.0-0)
├── yolov8n.pt              # Base YOLOv8n weights
├── Pothole.v1-raw.yolov8/  # Dataset
├── runs/                   # Training outputs and trained weights
└── Road_DetectAI_Project_Report.pdf
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/manoj333565/Pothole-and-Road-defect-Detector.git
cd Pothole-and-Road-defect-Detector
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

Make sure the trained model file `best.pt` exists at the path set in `app.py`.

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload a road image, adjust the confidence threshold if needed, and click **🚀 Run Road Damage Detection**.

---

## 🧠 Model Training

The model is YOLOv8n, trained on a pothole dataset.

| Setting | Value |
|---|---|
| Epochs | 30 |
| Image size | 416 |
| Batch size | 2 |
| Device | CPU |
| Early stopping patience | 8 |

Check the dataset paths in `Pothole.v1-raw.yolov8/data.yaml` before training, then run:

```bash
python train.py
```

---

## 📊 Model Evaluation

Results on the validation set (133 images, 330 pothole instances):

| Metric | Score |
|---|---:|
| Precision | **76.2%** |
| Recall | **63.3%** |
| mAP@50 | **72.1%** |
| mAP@50–95 | **43.7%** |

> Performance may vary on images the model hasn't seen.

---

## 📐 Severity Estimation

Damage coverage is calculated as:

```text
Coverage (%) = (Area covered by bounding boxes ÷ Image area) × 100
```

Overlapping boxes are merged before calculating coverage, so any area is counted only once.

| Coverage | Severity |
|---|---|
| No detections | 🟢 NONE |
| Less than 3% | 🟡 LOW |
| 3% to under 10% | 🟠 MEDIUM |
| 10% to under 20% | 🔴 HIGH |
| 20% or more | 🟣 CRITICAL |

> These are visual estimates based on image area, not measurements of real-world pothole dimensions.

---

## ⚠️ Limitations

- Small or poorly lit potholes may be missed.
- Road markings, shadows, and patches can cause false detections.
- Detection confidence is not the same as overall accuracy.
- Severity depends on image framing and bounding-box quality.
- This tool does not replace professional road inspection.

---

## 🔮 Future Improvements

- [ ] Improve recall on small and low-light potholes
- [ ] Add a confusion matrix and precision–recall curve
- [ ] Test on more road images and lighting conditions
- [ ] Add video and live-camera detection
- [ ] Deploy to Hugging Face Spaces

---

## 📄 Dataset and License

The dataset configuration lists its source license as **ODbL v1.0**. Check the dataset's license and attribution requirements before redistributing it.

---

<div align="center">

⭐ If you find this project useful, consider starring the repository.

</div>