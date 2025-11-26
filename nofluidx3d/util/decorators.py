# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Wed Nov 26 12:49:09 2025

@author: Richard Kellnberger
"""

import functools
import warnings


def deprecated(message: str):
    def decorator(fun):
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(
                f"Call to deprecated function {fun.__name__}: {message}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return fun(*args, **kwargs)

        return inner

    return decorator
