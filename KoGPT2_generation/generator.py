from model import KoGPT2IdeaModel

class KoGPT2IdeaGenerator():
    def __init__(self, args) -> None:
        self.model = KoGPT2IdeaModel.load_from_checkpoint(args.model_params)

    def generate(self, category_content):
        result = self.model.idea_maker(category_content)
        return result
        