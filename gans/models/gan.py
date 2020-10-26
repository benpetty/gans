import json
from uuid import uuid4

from gans.models.base import BaseModel


class GAN(BaseModel):

    id: str
    name: str

    def __init__(self, data: dict, **kwargs):
        """
        :param data: GAN data properties
        :type data: dict
        """
        self.id = data.get("id", str(uuid4()))
        self.name = data.get("name", None)

    def __repr__(self):
        og = super().__repr__()
        obj = {
            "id": self.id,
            "name": self.name,
        }
        return f"{og} {obj}"
