# 🚘 TireNET. | Tire Condition Monitoring System

## 🔎 About

This project focuses on tire condition classification using deep learning. The system can classify tire conditions into two categories: **Good** (safe to use) and **Defective** (requires replacement) using the MobileNetV2 with CBAM (Convolutional Block Attention Module) architecture.

Project languages:

- en

## 📦 Dependencies

| Name                                                     | Version |
| -------------------------------------------------------- | ------- |
| [torch](https://pypi.org/project/torch/)                 | 2.0.0+  |
| [torchvision](https://pypi.org/project/torchvision/)     | 0.15.0+ |
| [streamlit](https://pypi.org/project/streamlit/)         | 1.28.0+ |
| [Pillow](https://pypi.org/project/Pillow/)               | 10.0.0+ |
| [numpy](https://pypi.org/project/numpy/)                 | 1.24.0+ |
| [plotly](https://pypi.org/project/plotly/)               | 5.17.0+ |
| [opencv-python](https://pypi.org/project/opencv-python/) | 4.8.0+  |
| [scikit-learn](https://pypi.org/project/scikit-learn/)   | 1.3.0+  |

## 🖥️ Requirements

- Operating System (OS): Windows 10, MacOS, Linux
- `python>=3.9` and `pytorch>=2.0.0`
- CUDA 11.8+ (optional, for GPU acceleration)
- Integrated Development Environment (IDE): VSCode
- Web Browser: Google Chrome, Microsoft Edge, Firefox, Safari

## ⬇️ Installation

### Make a directory

```
mkdir tire_wear_classification
```

```
cd tire_wear_classification
```

### Create and activate environment

```
python -m venv venv
```

```
venv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

Or install manually:

```
pip install torch torchvision streamlit pillow numpy plotly opencv-python scikit-learn
```

### Run App

```
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## ✨ Features

- **Upload Image**: Upload tire photos for classification
- **Classification Result**: Display classification result with confidence score and visualization.

## 📁 Project Structure

```
tire_wear_classification/
├── app.py                          # Main Streamlit application
├── notebook.ipynb                  # Training notebook
├── requirements.txt                # Python dependencies
├── config/
│   └── train_config.json          # Model training configuration
├── models/
│   └── best_model.pth             # Trained model checkpoint
├── results/
│   ├── test_results.json          # Model test results & metrics
│   ├── training_history.json      # Training history
│   └── *.png                      # Result visualizations
├── assets/
│   └── banner.png                 # Banner image
├── utils/
│   ├── __init__.py
│   ├── model.py                   # Model architecture & loading
│   └── image_processor.py         # Image preprocessing & prediction
└── .streamlit/
    └── config.toml                # Streamlit configuration
```

## 🤖 How to Use

1. **Load Application**: Model loads automatically at startup
2. **View Model Info**: Check model specifications in sidebar
3. **Upload Image**: Click upload area or drag-drop tire image (JPG/PNG)
4. **View Results**:
   - Uploaded image displayed on the left
   - Classification result (Good/Defective) displayed on the right with confidence score
5. **Analyze Visualizations**:
   - Gauge chart shows confidence as a speedometer
   - Bar chart shows confidence comparison for all classes

## 🧠 Model Architecture

- **Model**: MobileNetV2 with CBAM Attention Mechanism
- **Input Size**: 224×224 pixels
- **Classes**:
  - 0: Good (Tire in optimal condition)
  - 1: Defective (Tire requires replacement)
  See details on `notebook.ipynb`

## 📊 Performance Metrics

### Overall

- Test Accuracy: **97.85%**
- Best Validation Accuracy: **97.12%**

### Good Class

- Precision: 96.83%
- Recall: 98.39%
- F1-Score: 97.60%

### Defective Class

- Precision: 98.69%
- Recall: 97.42%
- F1-Score: 98.05%

## 🥼 Author(s) / Contributor(s)

- Wicaksono Hanif Supriyanto

## 📚 References & Attribution

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv. https://arxiv.org/abs/1801.04381
- Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. arXiv. https://arxiv.org/abs/1807.06521
- P, PATHMANABAN; C, Abishek; Sai, Kousik muthayala; S, Karthick; S, Aakash (2023), "Digital images of defective and good condition tyres", Mendeley Data, V1, doi: 10.17632/bn7ch8tvyp.1
- Zhan, L., Xu, X., Qi, Q., Yan, Y., Wang, Y., Qian, F., & Zhu, N. (2023). An improved YOLOv7-based detection of tire defects for driving assistance. IFAC PapersOnLine, 56(2), 4904-4909.
