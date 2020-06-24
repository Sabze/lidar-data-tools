

class SequentialFeedbackCounter:
    def __init__(self, feedback_num:int):
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