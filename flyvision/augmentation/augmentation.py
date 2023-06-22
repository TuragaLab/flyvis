class Augmentation:
    augment = True

    def __call__(self, seq, *args, **kwargs):
        if self.augment:
            return self.transform(seq, *args, **kwargs)
        return seq

    def transform(self, seq, *args, **kwargs):
        return seq

    def set_or_sample(self, *args):
        pass
