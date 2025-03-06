from typing import Callable, Any

# a more functional if-then-else
ITE : Callable[[bool, Any, Any], Any] = lambda x, y, z :  y if x else z