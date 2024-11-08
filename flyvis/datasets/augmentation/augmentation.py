from typing import Any, Sequence


class Augmentation:
    """
    Base class for data augmentation operations.

    This class provides a framework for implementing various data augmentation
    techniques. Subclasses should override the `transform` method to implement
    specific augmentation logic.

    Attributes:
        augment (bool): Flag to enable or disable augmentation.
    """

    augment: bool = True

    def __call__(self, seq: Sequence[Any], *args: Any, **kwargs: Any) -> Sequence[Any]:
        """
        Apply augmentation to the input sequence if enabled.

        Args:
            seq: Input sequence to be augmented.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Augmented sequence if augmentation is enabled,
                otherwise the original sequence.
        """
        if self.augment:
            return self.transform(seq, *args, **kwargs)
        return seq

    def transform(self, seq: Sequence[Any], *args: Any, **kwargs: Any) -> Sequence[Any]:
        """
        Apply the augmentation transformation to the input sequence.

        This method should be overridden by subclasses to implement specific
        augmentation logic.

        Args:
            seq: Input sequence to be transformed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Transformed sequence.
        """
        return seq

    def set_or_sample(self, *args: Any) -> None:
        """
        Set or sample augmentation parameters.

        This method can be used to set fixed parameters or sample random
        parameters for the augmentation process.

        Args:
            *args: Arguments for setting or sampling parameters.
        """
        pass
