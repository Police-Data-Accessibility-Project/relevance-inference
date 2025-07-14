
def format_model_name_from_path(path: str) -> str:
    # Remove the `models` prefix
    model_name = path.split("models/")[1]
    # Replace slashes with double underscores
    model_name = model_name.replace("/", "__")
    return model_name
