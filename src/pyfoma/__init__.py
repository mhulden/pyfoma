from pyfoma.fst import *
from pyfoma.fst import re as regex
from pyfoma.paradigm import Paradigm
from pyfoma.algorithms import concatenate, union, intersection, kleene_star, kleene_plus, difference, cross_product, compose, optional, \
                                ignore, projected, inverted, reversed, reversed_e, minimized, minimized_as_dfa, determinized_as_dfa, determinized_unweighted, \
                                _algorithms_to_add


__author__     = "Mans Hulden"
__copyright__  = "Copyright 2022"
__credits__    = ["Mans Hulden"]
__license__    = "Apache"
__version__    = "2.0"
__maintainer__ = "Mans Hulden"
__email__      = "mans.hulden@gmail.com"
__status__     = "Prototype"


# This project is (partially) documented using the Sphinx docstring convention

# Dynamically add the algorithms contained in `algorithms.py` to the FST as instance methods
# Is this hacky? Yes, but it means we can keep our algorithm logic in the `algorithms` module,
# while still being able to call useful methods like `determinize` on the actual FST instances
# Note: These methods will be mutating, while the original versions are non-mutating
from inspect import signature, Signature

for mutating_name, original_func in _algorithms_to_add.items():
    original_signature = signature(original_func)

    # Use the original func to make a mutating version
    create_in_place_func = original_signature.return_annotation == 'FST'

    if create_in_place_func:
        new_func = (lambda orig_func: lambda self, *args, **kwargs: self.become(orig_func(self, *args, **kwargs)))(original_func)
    else:
        new_func = (lambda orig_func: lambda self, *args, **kwargs: orig_func(self, *args, **kwargs))(original_func)

    new_func.__name__ = mutating_name

    # Replace 'fst' or 'fst1' with 'self'
    self_param = original_signature.parameters.get('fst') or original_signature.parameters.get('fst1')
    if self_param:
        other_params = [original_signature.parameters[param] for param in original_signature.parameters if param not in ['fst', 'fst1']]
        original_signature = original_signature.replace(parameters=[self_param.replace(name='self')] + other_params)

    # If the method now modifies the FST in place and doesn't have a return, remove the return from the signature
    if create_in_place_func:
        if "Returns a modified FST" in original_func.__doc__:
            new_func.__doc__ = original_func.__doc__.replace("Returns a modified FST", "Mutates the FST")
        else:
            new_func.__doc__ = "Mutates the caller FST.\n" + original_func.__doc__
        new_func.__signature__ = original_signature.replace(return_annotation=Signature.empty)
    else:
        new_func.__doc__ = original_func.__doc__
        new_func.__signature__ = original_signature

    new_func.__annotations__ = original_func.__annotations__

    # Finally, add the new method to the FST class
    setattr(FST, mutating_name, new_func)


