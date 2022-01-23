import easydict
import os
import torch

# from idea_generation import generator
import idea_generation

# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())

if __name__ == "__main__":
    args = easydict.EasyDict({
        'gpus' : 1,
        'model_params' : 'idea_generation/model_chp/model_-last.ckpt'
    })
    evaluator = idea_generation.generator.KoGPT2IdeaGenerator(args)
    result = evaluator.generate("내구성")
    result_2 = evaluator.generate("디자인")
    print(result)
    print(result_2)

    r_list = evaluator.generate_nbest_ideas("디자인")
    r_list_2 = evaluator.generate_nbest_ideas("내구성")

    print(r_list)
    print(r_list_2)
