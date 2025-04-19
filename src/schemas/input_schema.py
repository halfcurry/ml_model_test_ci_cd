from pydantic import BaseModel

class Input_Schema(BaseModel):
    img_url: str