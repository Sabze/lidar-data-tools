
class SequentialFeedbackCounter:
    def __init__(self, feedback_num: int):
        self.feedback_num = feedback_num
        self.counter = 0
        print("Processing...", end='', flush=True)

    def count(self):
        self.counter += 1
        if self.counter == self.feedback_num:
            print(".", end='', flush=True)
            self.counter = 0

    def done(self):
        print()


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
