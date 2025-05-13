from fastapi import FastAPI, UploadFile, File, HTTPException, Body, APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

from config import config_options
from utils import (preprocess_single,
                   preprocess_multi_json,
                   preprocess_completion_json)

app = FastAPI(title="LMS Video Engagement Prediction Service")

# Load models at startup
models = {
    "single_feature": joblib.load(config_options["churn_single_feature"]["model_path"]),
    "multi_feature": joblib.load(config_options["churn_multi_feature"]["model_path"]),
    "completion_model": joblib.load(config_options["learner_completion prediction"]["model_path"])
}

# Pydantic schemas
class SingleRequest(BaseModel):
    completion_rate_percent: float

class MultiRequest(BaseModel):
    course_id: int
    video_title: str

class CompletionRequest(BaseModel):
    student_avg_completion: float
    course_id: int


# ---- Router for Churn Single Feature Prediction ----
router_churn_single = APIRouter(
    prefix="/predict/churn_single",
    tags=["Churn Single Feature Prediction"],
    responses={
        400: {"description": "Bad Request"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
    }
)

# ---- Router for Churn Multi‐Feature Prediction ----
router_churn_multi = APIRouter(
    prefix="/predict/churn_multi",
    tags=["Churn Multi Feature Prediction"],
    responses={
        400: {"description": "Bad Request"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
    },
)

# ---- Router for learner completion Prediction ----
router_completion = APIRouter(
    prefix="/predict/learner_completion",
    tags=["Learner Completion Prediction"],
    responses={400: {"description": "Bad Request"},
               422: {"description": "Validation Error"},
               500: {"description": "Internal Server Error"}}
)


@router_churn_single.post(
    "/json",
    summary="Predict churn (JSON only)",
    description="Accepts `{ \"completion_rate_percent\": <float> }` in the JSON body.",
)
async def predict_churn_single_json(payload: SingleRequest = Body(...)):
    model = models["single_feature"]
    try:
        X = [[payload.completion_rate_percent]]
        prob  = model.predict_proba(X)[0, 1]
        label = int(prob > 0.5)
        return {"churn_label": label}
        # return {"probability": prob, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router_churn_single.post(
    "/file",
    summary="Predict churn (CSV upload only)",
    description="Upload a CSV with column `completion_rate_percent`.",
)
async def predict_churn_single_file(
    file: UploadFile = File(..., description="CSV file")
):
    model = models["single_feature"]
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    if "completion_rate_percent" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="CSV must include a `completion_rate_percent` column."
        )

    try:
        X = df[["completion_rate_percent"]].values.tolist()
        probs  = model.predict_proba(X)[:, 1].tolist()
        labels = [int(p > 0.5) for p in probs]

        results = []
        for row, prob, label in zip(df.to_dict(orient="records"), probs, labels):
            row.update({
                "churn_probability": prob,
                "churn_label": label
            })
            results.append(row)

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# @router_churn_single.post(
#     "/file",
#     summary="Predict churn (CSV upload only)",
#     description="Upload a CSV with column `completion_rate_percent`.",
# )
# async def predict_churn_single_file(
#     file: UploadFile = File(..., description="CSV file")
# ):
#     model = models["single_feature"]

#     try:
#         df = pd.read_csv(file.file)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

#     if "completion_rate_percent" not in df.columns:
#         raise HTTPException(
#             status_code=422,
#             detail="CSV must include a `completion_rate_percent` column."
#         )

#     try:
#         X      = df[["completion_rate_percent"]].values.tolist()
#         probs  = model.predict_proba(X)[:, 1].tolist()
#         labels = [int(p > 0.5) for p in probs]
#         return {"churn_labels": labels}
#         # return {"probabilities": probs, "labels": labels}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@router_churn_multi.post(
    "/json",
    summary="Predict churn (JSON only)",
    description="Accepts `{ \"course_id\": <int>, \"video_title\": <str> }`.",
)
async def predict_churn_multi_json(payload: MultiRequest = Body(...)):
    model = models["multi_feature"]
    try:
        df = preprocess_multi_json(payload.dict())
        prob  = model.predict_proba(df)[:, 1][0]
        label = int(prob > 0.5)
        return {"probability": prob, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON prediction failed: {e}")

@router_churn_multi.post(
    "/file",
    summary="Predict churn (CSV upload only)",
    description="Upload a CSV with columns `course_id`, `video_title`.",
)
async def predict_churn_multi_file(
    file: UploadFile = File(..., description="CSV file containing course_id, video_title")
):
    model = models["multi_feature"]

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    missing = [c for c in ("course_id", "video_title") if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {missing}"
        )

    try:
        df["course_id_cat"]   = df["course_id"].astype(str)
        df["video_title_cat"] = df["video_title"].astype(str)
        X = df[["course_id_cat", "video_title_cat"]]
        probs  = model.predict_proba(X)[:, 1].tolist()
        labels = [int(p > 0.5) for p in probs]

        results = []
        for row, prob, label in zip(df.to_dict(orient="records"), probs, labels):
            row.update({
                "churn_probability": prob,
                "churn_label": label
            })
            results.append(row)

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {e}")


# @router_churn_multi.post(
#     "/file",
#     summary="Predict churn (CSV upload only)",
#     description="Upload a CSV with columns `course_id`, `video_title`.",
# )
# async def predict_churn_multi_file(
#     file: UploadFile = File(..., description="CSV file containing course_id, video_title")
# ):
#     model = models["multi_feature"]

#     try:
#         df_raw = pd.read_csv(file.file)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

#     missing = [c for c in ("course_id", "video_title") if c not in df_raw.columns]
#     if missing:
#         raise HTTPException(
#             status_code=422,
#             detail=f"CSV is missing required columns: {missing}"
#         )

#     try:
#         df_raw["course_id_cat"]   = df_raw["course_id"].astype(str)
#         df_raw["video_title_cat"] = df_raw["video_title"].astype(str)
#         df = df_raw[["course_id_cat", "video_title_cat"]]

#         probs  = model.predict_proba(df)[:, 1].tolist()
#         labels = [int(p > 0.5) for p in probs]
#         return {"probabilities": probs, "labels": labels}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"CSV prediction failed: {e}")


@router_completion.post(
    "/json",
    summary="Predict learner completion (JSON only)",
    description="Accepts JSON { student_avg_completion: float, course_id: int }"
)
async def predict_completion_json(payload: CompletionRequest = Body(...)):
    bundle = models['completion_model']
    enc = bundle['encoder']
    clf = bundle['model']
    try:
        X = preprocess_completion_json(payload.dict(), enc)
        prob  = clf.predict_proba(X)[:, 1][0]
        label = int(prob > 0.5)
        return {"probability": prob, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router_completion.post(
    "/file",
    summary="Predict learner completion (CSV upload)",
    description="Upload a CSV with columns `student_id`, `course_id`, and `completion_rate_percent`.",
)
async def predict_completion_file(
    file: UploadFile = File(..., description="CSV file containing student_id, course_id, completion_rate_percent")
):
    bundle = models["completion_model"]  
    enc   = bundle["encoder"]
    clf   = bundle["model"]

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    missing = [c for c in ("student_id", "course_id", "completion_rate_percent") if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    try:
        df["student_avg_completion"] = df.groupby("student_id")["completion_rate_percent"].transform("mean")
        df["course_id"] = df["course_id"].astype(str)
        course_ohe = pd.DataFrame(
            enc.transform(df[["course_id"]]),
            columns=enc.get_feature_names_out(["course_id"]),
            index=df.index
        )
        X = pd.concat([df[["student_avg_completion"]], course_ohe], axis=1)

        probs  = clf.predict_proba(X)[:, 1].tolist()
        labels = [int(p > 0.5) for p in probs]

        results = []
        for row, prob, label in zip(df.to_dict(orient="records"), probs, labels):
            row.update({
                "completion_probability": prob,
                "completion_label": label
            })
            results.append(row)

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# @router_completion.post(
#     "/file",
#     summary="Predict learner completion (CSV upload)",
#     description="Upload a CSV with columns `student_id`, `course_id`, and `completion_rate_percent`.",
# )
# async def predict_completion_file(
#     file: UploadFile = File(..., description="CSV file containing student_id, course_id, completion_rate_percent")
# ):
#     bundle = models['completion_model']   # {'encoder': enc, 'model': clf}
#     enc = bundle['encoder']
#     clf = bundle['model']

#     try:
#         df_raw = pd.read_csv(file.file)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

#     missing = [c for c in ("student_id", "course_id", "completion_rate_percent") if c not in df_raw.columns]
#     if missing:
#         raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

#     try:
#         df_raw['student_avg_completion'] = (
#             df_raw
#               .groupby('student_id')['completion_rate_percent']
#               .transform('mean')
#         )
#         df_raw['course_id'] = df_raw['course_id'].astype(str)
#         course_ohe = pd.DataFrame(
#             enc.transform(df_raw[['course_id']]),
#             columns=enc.get_feature_names_out(['course_id']),
#             index=df_raw.index
#         )
#         X = pd.concat([df_raw[['student_avg_completion']], course_ohe], axis=1)

#         probs  = clf.predict_proba(X)[:, 1].tolist()
#         labels = [int(p > 0.5) for p in probs]

#         return {"probabilities": probs, "labels": labels}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


# Routers
app.include_router(router_churn_single)
app.include_router(router_churn_multi)
app.include_router(router_completion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)