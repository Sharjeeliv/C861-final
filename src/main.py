
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from .utils import time_execution
from .data.encode import tfid_encode, seq_encode
from .data.data import get_datasets


@time_execution
def ml_process(X_tr, y_tr, X_te, y_te):
    
    X_tr, X_te = tfid_encode(X_tr, X_te)
    print("Training...")
    # model = LogisticRegression(max_iter=1000)
    model = LinearSVC()
    model.fit(X_tr, y_tr)
    
    print("Testing...")
    y_pred = model.predict(X_te)
    res = accuracy_score(y_te, y_pred)
    print(f"Accuracy: {res}")


def dl_preocess(X_tr, y_tr, X_te: pd.Series, y_te):
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    
    for labels, texts in tr_loader:
        print(f"Batch Labels Shape: {labels.shape}")
        print(f"Batch Texts Shape: {texts.shape}")
        break
    

def test():
    X_tr, y_tr, X_te, y_te = get_datasets()
    dl_preocess(X_tr, y_tr, X_te, y_te)


if __name__ == "__main__":
    test()
