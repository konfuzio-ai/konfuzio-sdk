"""Schemas for validating the structure of Konfuzio-related objects."""
from typing import List, Union

from pydantic import BaseModel, Field


class CategoriesLabelData20240409(BaseModel):
    """Describe a schema for the file containing Category and Label information."""

    class CategorySchema(BaseModel):
        """Describe a schema for the Category class on 09/04/2024."""

        class LabelSetSchema(BaseModel):
            """Describe a schema for the Label Set class on 09/04/2024."""

            class LabelSchema(BaseModel):
                """Describe a schema for the Label class on 09/04/2024."""

                name: str
                data_type: str
                has_multiple_top_candidates: bool
                id: int
                api_name: str
                threshold: float

            api_name: str
            has_multiple_annotation_sets: bool
            id: int
            labels: List[LabelSchema]

        api_name: str
        name: str
        id: int
        project: Union[int, None]
        schema_: List[LabelSetSchema] = Field(alias='schema')

    categories: List[CategorySchema]

    class Config:
        arbitrary_types_allowed = True
