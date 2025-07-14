from typing import Dict, Any

import spacy
from environs import Env
from huggingface_hub import hf_hub_download
from joblib import load

from src.dtos.output.basic import BasicOutput

from src.format import format_model_name_from_path
from src.models.bag_of_words.extractor import BagOfWordsExtractor
from src.models.bag_of_words.formatter import BagOfWordsFormatter
from src.models.bag_of_words.model import BagOfWordsModelContainer
from src.models.bag_of_words.predictor import RelevancePredictor

SPACY_MODEL = spacy.load('en_core_web_trf', disable=['parser'])  # Largest, slowest, most accurate model


class EndpointHandler:
    def __init__(self, path: str):
        env = Env()
        env.read_env()

        model_path = env.str("MODEL_PATH")
        self.model_name = format_model_name_from_path(model_path)
        downloaded_model_path = hf_hub_download(
            repo_id="PDAP/url-relevance-models",
            subfolder=model_path,
            filename="model.joblib"
        )
        self.model_container: BagOfWordsModelContainer = load(downloaded_model_path)
        self.extractor = BagOfWordsExtractor(self.model_container.permitted_terms)
        self.formatter = BagOfWordsFormatter(self.model_container.term_label_encoder)
        self.predictor = RelevancePredictor(self.model_container.model)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        html = inputs["inputs"]
        bag_of_words = self.extractor.extract_bag_of_words(html)
        csr = self.formatter.format_bag_of_words(bag_of_words)
        output = self.predictor.predict_relevance(csr)
        return BasicOutput(
            annotation=output.is_relevant,
            confidence=output.probability,
            model=self.model_name
        ).model_dump(mode="json")
