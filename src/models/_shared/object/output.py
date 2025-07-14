from pydantic import BaseModel


class PredictionOutput(BaseModel):
    is_relevant: bool
    probability: float