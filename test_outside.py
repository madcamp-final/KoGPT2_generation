import easydict
import os
import torch

# from idea_generation import generator
import idea_generation

if __name__ == "__main__":
    args = easydict.EasyDict({
        'gpus' : 1,
        'model_params' : 'idea_generation/model_chp/model_-last.ckpt'
    })
    evaluator = idea_generation.generator.KoGPT2IdeaGenerator(args)
    result = evaluator.generate("디자인")
    result_2 = evaluator.generate("내구성")
    result_3 = evaluator.generate("상태-미개봉")
    result_4 = evaluator.generate("상태-거의 새 것")
    result_5 = evaluator.generate("상태-사용감 있음")
    print(result)
    print(result_2)
    print(result_3)
    print(result_4)
    print(result_5)

    r_list = evaluator.generate_nbest_ideas("디자인")
    r_list_2 = evaluator.generate_nbest_ideas("내구성")
    r_list_3 = evaluator.generate_nbest_ideas("상태-미개봉")
    r_list_4 = evaluator.generate_nbest_ideas("상태-거의 새 것")
    r_list_5 = evaluator.generate_nbest_ideas("상태-사용감 있음")
    print(r_list)
    print(r_list_2)
    print(r_list_3)
    print(r_list_4)
    print(r_list_5)

    result_temp = evaluator.generate_temp_ideas("디자인")
    result_temp_2 = evaluator.generate_temp_ideas("내구성")
    result_temp_3 = evaluator.generate_temp_ideas("상태-미개봉")
    result_temp_4 = evaluator.generate_temp_ideas("상태-거의 새 것")
    result_temp_5 = evaluator.generate_temp_ideas("상태-사용감 있음")


