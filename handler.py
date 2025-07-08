from typing import Dict, Any

import en_core_web_trf
from environs import Env
from huggingface_hub import hf_hub_download
from joblib import load

SPACY_MODEL = en_core_web_trf.load()



class EndpointHandler:
    def __init__(self, path: str):
        env = Env()
        env.read_env()

        model_path = env.str("MODEL_PATH")
        downloaded_model_path = hf_hub_download(
            repo_id="PDAP/url-relevance-models",
            subfolder=model_path,
            filename="model.joblib"
        )
        self.model = load(downloaded_model_path)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # Expecting input like: {"inputs": "<html>...</html>"}
        html = inputs["inputs"]
        return {"label": str(self.model)}
