class NoFinalStatesException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg or "FST has no final states!")
