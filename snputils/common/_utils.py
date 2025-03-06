from typing import Callable, Any
# don't call this. This is for us to use. 
# a more functional if-then-else
ITE : Callable[[bool, Any, Any], Any] = lambda x, y, z :  y if x else z