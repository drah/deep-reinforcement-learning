from networks import Actor

class Algorithm:
    def __init__(self, actor: Actor, **kwargs):
        raise NotImplementedError("Don't call me.")

    def run(self, **kwargs):
        raise NotImplementedError("Don't call me.")
