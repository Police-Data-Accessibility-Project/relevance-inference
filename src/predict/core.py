from scipy.sparse import csr_matrix

from src.predict.output import PredictionOutput
from types_ import FitPredictor


class RelevancePredictor:

    def __init__(
        self,
        model: FitPredictor
    ):
        self.model = model

    def predict_relevance(self, csr: csr_matrix) -> PredictionOutput:
        y_pred = self.model.predict(csr)
        probability_estimates = self.model.predict_proba(csr)[:, 1]
        return PredictionOutput(
            is_relevant=y_pred[0] == 1,
            probability=probability_estimates[0]
        )