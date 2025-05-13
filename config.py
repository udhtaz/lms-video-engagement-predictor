
config_options = {
    "churn_single_feature": {
        "model_path": "models/baselineLR_churn_model.joblib", 
        "input_type": "single", 
    },
    "churn_multi_feature": {
        "model_path": "models/RF_churn_model.joblib", 
        "input_type": "multi", 
    },
    "learner_completion prediction": {
        "model_path": "models/learner_completion_model.joblib", 
        "input_type": "multi", 
    }
}