from pydantic import BaseModel, Field, confloat


class BasicOutput(BaseModel):
    annotation: bool = Field(
        description="Whether the annotation is relevant or not"
    )
    confidence: float = Field(
        description="The confidence of the prediction",
        ge=0,
        le=1
    )
    model: str = Field(
        # Characters with underscores, followed by two underscores, followed by a timestamp
        pattern="^\w+__[\d\-\_]+$",
        description="The model used to make the prediction"
    )