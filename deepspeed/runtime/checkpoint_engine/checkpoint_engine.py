class CheckpointEngine(object):

    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        pass

    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def save(self, state_dict, path: str):
        pass

    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        # to tell checkpoint services if all files are readys.
        pass
