from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class BagOfWordsModelContainer(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model: LogisticRegression
    term_label_encoder: LabelEncoder
    permitted_terms: list[str]
