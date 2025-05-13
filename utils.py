import pandas as pd
import joblib
from sklearn import __version__

def preprocess_single(data: dict) -> list:
    """
    Preprocess a single JSON payload for the single-feature model.
    Expects: {"completion_rate_percent": float}
    Returns: [[completion_rate_percent]]
    """
    return [data["completion_rate_percent"]]


def preprocess_multi_json(data: dict) -> pd.DataFrame:
    """
    Preprocess a single JSON payload for the multi-feature model.
    Expects: {"course_id": int, "video_title": str}
    Returns a DataFrame with encoded columns.
    """
    df = pd.DataFrame([data])
    df["course_id_cat"] = df["course_id"].astype(str)
    df["video_title_cat"] = df["video_title"].astype(str)
    return df[["course_id_cat","video_title_cat"]]


def preprocess_multi_csv(file) -> pd.DataFrame:
    """
    Preprocess a CSV file upload for the multi-feature model.
    CSV must contain columns: course_id, video_title
    Returns a DataFrame with encoded columns.
    """
    df = pd.read_csv(file)
    df["course_id_cat"] = df["course_id"].astype(str)
    df["video_title_cat"] = df["video_title"].astype(str)
    return df[["course_id_cat","video_title_cat"]]


def preprocess_completion_json(data: dict, encoder) -> pd.DataFrame:
    """
    Preprocess JSON input for learner completion model.
    Expects keys:
      - student_avg_completion: float
      - course_id: int

    Returns a DataFrame ready for prediction, with one-hot encoded course features.
    """
    df = pd.DataFrame([{
        'student_avg_completion': data['student_avg_completion'],
        'course_id': str(data['course_id'])
    }])
    course_ohe = pd.DataFrame(
        encoder.transform(df[['course_id']]),
        columns=encoder.get_feature_names_out(['course_id']),
        index=df.index
    )
    return pd.concat([df[['student_avg_completion']], course_ohe], axis=1)


def upgrade_model(input_path, output_path):
    """
    Load a model saved with an older version of scikit-learn and save it again
    with the current version to ensure compatibility.

    Args:
        input_path (str): Path to the old model file.
        output_path (str): Path to save the upgraded model.
    """
    print(f"Current scikit-learn version: {__version__}")
    model = joblib.load(input_path)
    joblib.dump(model, output_path)
    print(f"Model upgraded and saved to {output_path}")