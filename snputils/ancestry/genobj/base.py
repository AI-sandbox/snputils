import abc
import copy

from snputils._utils.printing import format_repr


class AncestryObject(abc.ABC):
    """
    Abstract class for ancestry data.
    """
    def __init__(self, n_samples, n_ancestries) -> None:
        self.__n_samples = n_samples
        self.__n_ancesties = n_ancestries

    def __repr__(self) -> str:
        return format_repr(
            self,
            shape=self.shape,
            n_samples=self.n_samples,
            n_ancestries=self.n_ancestries,
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def n_ancestries(self) -> int:
        """
        Retrieve `n_ancestries`.

        Returns:
            int: The total number of unique ancestries.
        """
        return self.__n_ancesties

    @property
    def n_samples(self) -> int:
        """
        Retrieve number of samples

        Returns:
            int: number of sample in the data.
        """
        return self.__n_samples

    @property
    def shape(self) -> tuple[int, int]:
        """
        Retrieve the primary ancestry matrix shape.

        Returns:
            tuple: `(n_samples, n_ancestries)`.
        """
        return (self.n_samples, self.n_ancestries)

    @property
    def copy(self):
        """
        Create a copy of the Ancestry Object

        Returns:
            dict: new Ancestry Object being a copy of the original Ancestry Object.
        """
        return copy.deepcopy(self)

    @abc.abstractmethod
    def _sanity_check(self) -> None:
        return
