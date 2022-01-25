from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

from idea_generation import generator
import easydict
import random

app = FastAPI()

args = easydict.EasyDict({
    'gpus' : 1,
    'model_params' : 'idea_generation/model_chp/model_-last.ckpt',
    'batch-size' :32
})
idea_generator = generator.KoGPT2IdeaGenerator(args)


class RequestCategory(BaseModel):
    #상태
    category_content_1: str
    #디자인
    category_content_2: str
    #내구성
    category_content_3: str

def idea_filter(category):
    result = []
    g_result = idea_generator.generate(category)
    gn_result = idea_generator.generate_nbest_ideas(category)
    gt_result = idea_generator.generate_temp_ideas(category)
    result.append(g_result)
    result.append(gt_result)
    while 1:
        random.shuffle(gn_result)
        result.append(gn_result[0])
        result.append(gn_result[1])
        result.append(gn_result[2])
        list(set(result))
        random.shuffle(gn_result)
        while 1:
            if(len(result)>3):
                del result[0]
            if(len(result)==3):
                break
        if(len(result)==3):
            break
    return result


@app.post("/idea_generation/")
async def idea_generate(item: RequestCategory):
    result1 = idea_filter(item.category_content_1)
    result2 = idea_filter(item.category_content_2)
    result3 = idea_filter(item.category_content_3)

    final_result = jsonable_encoder({item.category_content_1: result1, item.category_content_2: result2, item.category_content_3: result3})
    print(final_result)

    return final_result

