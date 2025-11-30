from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_absolute_error, accuracy_score

from .data import get_datasets
from .utils import time_execution

@time_execution
def ml_process(X_tr, y_tr, X_te, y_te):
    
    print("Encoding...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_tr)
    X_test_tfidf = tfidf_vectorizer.transform(X_te)

    print("Training...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_tr)
    
    print("Testing...")
    y_pred = model.predict(X_test_tfidf)
    res = accuracy_score(y_te, y_pred)
    print(f"Accuracy: {res}")
    # print(y_pred)

    # res = mean_absolute_error(y_te, y_pred)
    # print(f"MAE: {res}")
    


def test():
    X_tr, y_tr, X_te, y_te = get_datasets()
    ml_process(X_tr, y_tr, X_te, y_te)
    


if __name__ == "__main__":
    test()
