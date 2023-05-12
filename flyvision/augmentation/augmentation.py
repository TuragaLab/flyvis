import torch


class Augmentation:
    _augment = True
    _call = None

    def __call__(self, *args, **kwargs):
        assert isinstance(args[0], torch.Tensor)
        return args[0]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._call = cls.__call__

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value

        cls = type(self)
        if cls != Augmentation and Augmentation in cls.__mro__:
            if self.augment:
                cls.__call__ = cls._call
            else:
                # turn off augmentation by overiding subclass call to identity function
                cls.__call__ = Augmentation.__call__

    def set_or_sample(self, *args):
        pass
