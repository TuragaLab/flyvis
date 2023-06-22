import abc


class Augmentation(metaclass=abc.ABCMeta):
    augment = True

    def __call__(self, seq, *args, **kwargs):
        if self.augment:
            return self.transform(seq, *args, **kwargs)
        return seq

    @abc.abstractproperty
    def transform(self, seq, *args, **kwargs):
        pass

    @abc.abstractproperty
    def set_or_sample(self, *args):
        pass
