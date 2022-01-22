import easydict

# from idea_generation import generator
import idea_generation

if __name__ == "__main__":
    args = easydict.EasyDict({
        'gpus' : 1,
        'model_params' : 'idea_generation/model_chp/model_-last.ckpt'
    })
    evaluator = idea_generation.generator.KoGPT2IdeaGenerator(args)
    result = evaluator.generate("내구성")
    print(result)
