"""
The model families the pipeline trains, and how a trained model is logged to MLflow.

Every model is logged in MLflow's scikit-learn format with predict_proba. Whichever model wins, the
served model answers the same way: for each wine, [chance it is not high quality, chance it is].
"""
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# MLflow 3 saves scikit-learn models with skops instead of pickle, because loading a pickle can run any
# code. skops refuses the types it can't vouch for unless they are listed as trusted. These are the
# reviewed types each family needs, and the pipeline only loads models it trained itself.
TRUSTED_TYPES = {
    "logistic_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
    "xgboost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
}


def build_model(family, params, seed):
    """An untrained model of this family, with these settings."""
    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**params, random_state=seed))
    if family == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    if family == "xgboost":
        # A fixed number of threads, so the results don't depend on how many cores the computer has
        return XGBClassifier(**params, random_state=seed, n_jobs=4)
    raise ValueError(f"Unknown model family {family!r}: use one of {list(TRUSTED_TYPES)}")


def log_model(model, family, inputs):
    """
    Logs a trained model to the current run, with its input schema (signature) and an input example.
    Returns MLflow's ModelInfo, whose model_uri is what gets registered.
    """
    signature = infer_signature(inputs, model.predict_proba(inputs))
    return mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=inputs.head(5),
                                    pyfunc_predict_fn="predict_proba", skops_trusted_types=TRUSTED_TYPES[family])
