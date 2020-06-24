class NumberDict:
    """Dictionary with digits as items. """
    def __init__(self):
        self.number_dict = {}

    def update(self, other_dict: dict):
        for key, item in other_dict.items():
            self.number_dict[key] = self.number_dict.get(key, 0) + item

    def add(self, keys: list, items: list):
        for key, item in zip(keys, items):
            self.number_dict[key] = self.number_dict.get(key, 0) + item

