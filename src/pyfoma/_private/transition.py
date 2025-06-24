class Transition:
    __slots__ = ['targetstate', 'label', 'weight']
    def __init__(self, targetstate, label, weight):
        self.targetstate = targetstate
        self.label = label
        self.weight = weight
