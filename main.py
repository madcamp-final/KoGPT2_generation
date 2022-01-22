from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from idea_generation import generator
import easydict

app = FastAPI()

args = easydict.EasyDict({
    'gpus' : 1,
    'model_params' : 'idea_generation/model_chp/model_-last.ckpt'
})
idea_generator = generator.KoGPT2IdeaGenerator(args)


class Category(BaseModel):
    category_content: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/idea_generation/")
async def idea_generate(item: Category):
    item_dict = item.dict()
    result = idea_generator.generate(item.category_content)
    return result

