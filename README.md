# VisDrone object detection and Comparative Study with YOLO 

This project uses **YOLO** on the **VisDrone** dataset. I train several models, look at loss and results, try a bigger model and then L2 regularization, and compare a few different YOLO versions.

---

## What is in this repo

| File or folder | What it is for |
|----------------|----------------|
| [`Preprocessing_VisDrone.ipynb`](Preprocessing_VisDrone.ipynb) | Getting the data ready for YOLO |
| [`baseline-model.ipynb`](baseline-model.ipynb) | Train the **YOLOv11n** baseline |
| [`experimental-design.ipynb`](experimental-design.ipynb) | Train **YOLOv11s** (bigger model) |
| [`iterative-model.ipynb`](iterative-model.ipynb) | Train **YOLOv11s** with **L2** (weight decay) |
| [`comparison-model1.ipynb`](comparison-model1.ipynb) … [`comparison-model3.ipynb`](comparison-model3.ipynb) | Train **YOLOv26n**, **YOLOv5n**, **YOLOv8n** |
| [`Analysis/`](Analysis/) | Notebooks that **explain results** (loss curves, overfitting, plots) |
| [`Analysis/results-discussion-comparison.ipynb`](Analysis/results-discussion-comparison.ipynb) | **All models in one place**: tables + confusion matrices + PR curves |
| [`test-challenge-evaluation.ipynb`](test-challenge-evaluation.ipynb) | **Course competition:** runs final `best.pt` on **VisDrone test-challenge** images and save predictions |
| `*_export/` | Saved results: `results.csv`, images like `results.png`, `confusion_matrix.png`, `BoxPR_curve.png` |

---

## What I did (short summary)

I built a drone-view object detection project on VisDrone with Ultralytics YOLO.

1. I started with YOLOv11n as a baseline.  
2. I trained YOLOv11s to see if a larger model helps on the same data.  
3. The train and validation loss were farther apart for the bigger model, so I tried stronger L2 (weight decay) on YOLOv11s to help generalization.  
4. I also trained YOLOv26n, YOLOv5n, and YOLOv8n with the same kind of setup so I can compare different YOLO families.

The `Analysis/` notebooks show training vs validation loss, mAP, and my notes on convergence, overfitting, underfitting, and how dataset size and model size matter.

The [comparison notebook](Analysis/results-discussion-comparison.ipynb) puts numbers in tables (mAP, precision, recall, F1, rough model size, training time) and shows confusion matrices and PR curves for every run next to each other.

---

## Where to read about loss curves and fitting

Each analysis notebook matches one training run:

| What | Notebook |
|------|----------|
| Baseline YOLOv11n | [`Analysis/baseline-model-analysis.ipynb`](Analysis/baseline-model-analysis.ipynb) |
| Bigger model YOLOv11s | [`Analysis/experimental-design-analysis.ipynb`](Analysis/experimental-design-analysis.ipynb) |
| YOLOv11s + L2 | [`Analysis/iterative-model-analysis.ipynb`](Analysis/iterative-model-analysis.ipynb) |
| YOLOv26n | [`Analysis/comparison1-model-analysis.ipynb`](Analysis/comparison1-model-analysis.ipynb) |
| YOLOv5n | [`Analysis/comparison2-model-analysis.ipynb`](Analysis/comparison2-model-analysis.ipynb) |
| YOLOv8n | [`Analysis/comparison3-model-analysis.ipynb`](Analysis/comparison3-model-analysis.ipynb) |

---

## Order of the experiments

1. **YOLOv11n**: baseline on VisDrone.  
2. **YOLOv11s**: same idea, more parameters.  
3. **YOLOv11s + L2**: after seeing the train/val gap, I added more regularization. The comparison notebook has before and after numbers.  
4. **YOLOv26n, YOLOv5n, YOLOv8n**: compare different YOLO versions on the same general setup.

---

## How to reuse this repo

I trained mostly in Kaggle / Colab-style notebooks. Each run folder has an `args.yaml` with the settings I used.

You can clone the repo and open the notebooks. You do not have to re-train to read the analysis or the comparison notebook.

To run [`Analysis/results-discussion-comparison.ipynb`](Analysis/results-discussion-comparison.ipynb), start Jupyter from the main project folder (`yolo`) or from `Analysis/`, so the paths to `*_export/` work.

---

## References

- **VisDrone** dataset  
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** (how mAP, precision, and recall are defined)

This repo is for **CMPE 401** (object detection project).
