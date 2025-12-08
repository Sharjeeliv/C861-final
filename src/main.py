
# Local
from .utils.utils import time_execution
from .data.encode import tfid_encode, seq_encode
from .data.data import get_datasets, get_val_split
from .models import MODELS
from .train import tune, test, _metrics, _pack_metrics, P_ROOT, extract_features
from .utils.utils import save_output

# Builtin

# External
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# ********************************
# VARIABLES AND SETUP
# ********************************
ML_MODELS = {
    'LR': LogisticRegression,
    'SVC': LinearSVC
}


# ********************************
# HELPER FUNCTIONS
# ********************************
def ml_loop(X_tr, y_tr, X_te, y_te, title=None):
    for k, v in ML_MODELS.items():
        name = f"{title}-{k}" if title else k
        print(f'Training: {k}')
        
        model = ML_MODELS[k]()
        if isinstance(model, LogisticRegression): model.set_params(max_iter=1000)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_te)
        a, p, r, f = _metrics(y_te, y_pred)
        
        print(f"Accuracy: {a:.4}")
        res = _pack_metrics(a, p, r, f)
        save_output(res, {}, name, P_ROOT)

@time_execution
def expr_1(run_all=False, unique=False):    
    # Load and prepare datasets
    X_tr, y_tr, X_te, y_te = get_datasets(run_all=run_all, unique=unique)
    X_tn, X_va, y_tn, y_va = get_val_split(X_tr, y_tr)
    
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    tn_loader, va_loader = seq_encode(X_tn, y_tn, X_va, y_va, val=True)
    
    all_models = MODELS.keys()
        
    flag = "u" if unique else 'a'
    for model_name in all_models:
        title = f"{flag}_{model_name}"
        # Standard Experiment
        print(f'Tuning: {model_name}')
        params = tune(model_name, tn_loader, va_loader)
        print(f'Testing: {model_name}')
        model, res = test(model_name, params, tr_loader, te_loader)
        save_output(res, params, f"{title}", P_ROOT)
        
        # Additiona Experiments
        print(f"Extracting features for hybrid LR: {model_name}")
        X_tr1, y_tr1 = extract_features(model, tr_loader)
        X_te1, y_te1 = extract_features(model, te_loader)
        
        # ML Model Experiments
        ml_loop(X_tr1, y_tr1, X_te1, y_te1, title)


def expr_2(run_all=False, unique=False):    
    # Load and prepare datasets
    X_tr, y_tr, X_te, y_te = get_datasets(run_all=run_all, unique=unique)
    X_tr, X_te = tfid_encode(X_tr, X_te)
    ml_loop(X_tr, y_tr, X_te, y_te)
    

# ********************************
# MAIN FUNCTION
# ********************************
if __name__ == "__main__":
    expr_1(unique=True)
    expr_2(unique=True)
