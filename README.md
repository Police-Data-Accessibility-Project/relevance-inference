# relevance-inference

A machine learning model to be synced to Hugging Face. For use in their Inference API to evaluate whether a URL is relevant.

# Common Files

- *model.py* - model container
- *extractor.py* - extracts relevant data from handler input
- *formatter.py* - formats data for model ingestion
- *predictor.py* - predicts relevance based on formatted data 