# Mood-Detector
An end-to-end emotion classification pipeline: from raw FER2013 data to containerized REST API, training and evaluation reproduction using DVC and expermentation tracking using MLFlow. 

## DEMO
<img src="assets/demo.gif" width="700" alt="Mood Detector Demo UI">

`docker build -t mood-detector:v1 .`

`docker run -p 8000:8000 mood-detector:v1`

then open http://localhost:8000 in your browser

## Architecture Overview
**Pipeline Diagram:**

```mermaid
flowchart TD
    A[FER2013 Dataset\nKaggle API] --> B[data_ingestion.py\nDownload + Extract]
    B --> C[Preprocessing\nDrop classes 1,2,5,6\nRemap 0→0, 3→1, 4→2]
    C --> D[Train / Val / Test Split\n80% / 10% / 10%]
    D --> E[(data/processed/\ntrain.pkl\nval.pkl\ntest.pkl)]
    E --> F[model_training.py\nResidual CNN\nAdam + CrossEntropyLoss\nMLflow Tracking]
    F --> G[(model/\nbest_emotion_model.pth\nnormalization_stats.json\nclass_weights_stats.json\nrun_info.json)]
    G --> H[model_evaluation.py\nTest Set Evaluation\nClassification Report]
    H --> I[(reports/\neval_metrics.json)]
    H --> J[(mlflow.db\nTest Metrics\nArtifacts)]
    F --> J

    style A fill:#4A90D9,color:#fff
    style E fill:#2ECC71,color:#fff
    style G fill:#2ECC71,color:#fff
    style I fill:#2ECC71,color:#fff
    style J fill:#E67E22,color:#fff
```

---

**Serving Diagram:**

```mermaid
flowchart LR
    A[User Uploads\nJPEG / PNG] --> B[FastAPI\n/predict]
    B --> C[Read Image Bytes\nDecode via OpenCV]
    C --> D[Haar Cascade\nFace Detection]
    D -->|No face found| E[Return\n404 message]
    D -->|Face ROI| F[Preprocess\nGrayscale → Resize 48x48\nToTensor → Normalize]
    F --> G[emotionModel\nResidual CNN\n3-class output]
    G --> H[Softmax\nPer-class probabilities]
    H --> I[JSON Response\nemotion\nconfidence\nprobabilities]

    B2[FastAPI\n/health] --> J[Return\nstatus: healthy\nmodel_loaded: true]

    style A fill:#4A90D9,color:#fff
    style E fill:#E74C3C,color:#fff
    style I fill:#2ECC71,color:#fff
    style J fill:#2ECC71,color:#fff
    style G fill:#9B59B6,color:#fff
```

## MLflow Experiment Results

**Metrics across multiple runs**

| Metrics | RUN 1 | RUN 2 | RUN 3 | RUN 4 | RUN 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Best Validation F1 Score | 0.759 | 0.676 | 0.745 | 0.735 | 0.751 |
| Train Accuracy | 89.94 | 90.42 | 95.05 | 93.02 | 89.45 |
| Train Loss | 0.275 | 0.266 | 0.135 | 0.191 | 0.271 |
| Validation Accuracy | 71.95 | 66.26 | 67.24 | 73.43 | 74.56 |
| Validaton Loss | 0.637 | 0.938 | 1.043 | 0.695 | 0.608 |
| Test Accuracy | 78.49 | 69.17 | 74.13 | 75.78 | 78.21 |
| Test Loss | 0.582 | 0.713 | 0.612 | 0.616 | 0.587 |
| Test F1 Score | 0.756 | 0.675 | 0.728 | 0.74 | 0.755 |