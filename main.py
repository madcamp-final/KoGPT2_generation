from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

from idea_generation import generator
import easydict

app = FastAPI()

args = easydict.EasyDict({
    'gpus' : 1,
    'model_params' : 'idea_generation/model_chp/model_-last.ckpt',
    'batch-size' :32
})
idea_generator = generator.KoGPT2IdeaGenerator(args)


class RequestCategory(BaseModel):
    category_content_1: str
    category_content_2: str
    category_content_3: str


@app.post("/idea_generation/")
async def idea_generate(item: RequestCategory):
    result_1 = idea_generator.generate(item.category_content_1)
    result_2 = idea_generator.generate(item.category_content_2)
    result_3 = idea_generator.generate(item.category_content_3)

    final_result = jsonable_encoder({item.category_content_1: result_1, item.category_content_2: result_2, item.category_content_3: result_3})
    print(final_result)

    return final_result

