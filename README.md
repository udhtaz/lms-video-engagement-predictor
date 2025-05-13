# LMS Video Engagement Prediction

This project provides a FastAPI-based service for predicting:
1. **Churn Prediction**: Whether a user will churn based on single (`completion_rate_percent`) or multiple features (`course_id` and `video_title`).
2. **Video Completion Prediction**: Whether a student will complete a video based on their past completion rate and course.

## Features
- **Endpoints**:
  - `/predict/single`: Predict churn using a single feature (`completion_rate_percent`).
  - `/predict/multi`: Predict churn using multiple features (`course_id` and `video_title`).
  - `/predict/completion`: Predict video completion based on past completion rate and course.
- **Models**:
  - Single-feature Logistic Regression.
  - Multi-feature Random Forest.
  - Video Completion Prediction Model.

## Project Structure
```
LMS_Video_Engagement/
├── app.py                # FastAPI application
├── config.py             # Configuration for model paths
├── Dockerfile            # Dockerfile for containerization
├── init.sh               # Script to build and run the Docker container
├── requirements.txt      # Python dependencies
├── utils.py              # Preprocessing utilities
├── models/               # Directory containing trained models
│   ├── baselineLR_churn_model.joblib
│   ├── learner_completion_model.joblib
│   └── RF_churn_model.joblib
├── notebook/             # Data and analysis notebooks
│   ├── LMS_Engagement_Analysis.ipynb
│   ├── LMS Video Engagement Data.csv
│   └── video_stats.csv
```

## Setup

### Prerequisites
- Python 3.12+
- Docker

### Installation
Create venv
   ```bash
python 3.12 -m venv venv
   ```

Activate venv
   ```bash
source venv/bin/activate
   ```
   
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Locally
Run the FastAPI application:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Using Docker
1. Build and run the Docker container using the `init.sh` script:
   ```bash
   ./init.sh
   ```
2. Access the API at `http://localhost:80`.

## Endpoints
- **`/predict/single`**:
  - Input: JSON or CSV with `completion_rate_percent`.
  - Output: Churn probability and label.
- **`/predict/multi`**:
  - Input: JSON or CSV with `course_id` and `video_title`.
  - Output: Churn probabilities and labels.
- **`/predict/completion`**:
  - Input: JSON with `past_completion_rate` and `course_id`.
  - Output: Completion probability and label.

## Models
- Models are stored in the `models/` directory:
  - `baselineLR_churn_model.joblib`: Single-feature Logistic Regression.
  - `RF_churn_model.joblib`: Multi-feature Random Forest.
  - `learner_completion_model.joblib`: Video Completion Prediction Model.

## License
This project is licensed under the Apache License.
