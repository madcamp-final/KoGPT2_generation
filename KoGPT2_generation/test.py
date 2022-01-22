import easydict

from generator import KoGPT2IdeaGenerator

if __name__ == "__main__":
    args = easydict.EasyDict({
        'gpus' : 1,
        'model_params' : 'model_chp/model_-last.ckpt'
    })
    evaluator = KoGPT2IdeaGenerator(args)
    result = evaluator.generate("내구성")
    print(result)
