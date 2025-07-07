from typing import Dict, Any
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



class EndpointHandler:
    def __init__(self, path: str):
        # model_dir = os.getenv("HF_MODEL_DIR", ".")
        #
        # with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        #     self.model = pickle.load(f)
        #
        # # optional: you could also load a vocabulary or vectorizer
        # with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
        #     self.vectorizer = pickle.load(f)

        # 1. Generate synthetic binary classification data
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

        # 2. Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Create and train the Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)



    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # Expecting input like: {"inputs": "<html>...</html>"}
        html = inputs["inputs"]
        return {"label": str(1)}
