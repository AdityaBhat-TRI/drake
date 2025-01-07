"""Provides extensions for containers of Drake-related objects."""

import numpy as np
import re


class _EqualityProxyBase:
    # Wraps an object with a non-compliant `__eq__` operator (returns a
    # non-bool convertible expression) with a custom compliant `__eq__`
    # operator.
    def __init__(self, value):
        self._value = value

    def _get_value(self):
        return self._value

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        raise NotImplemented("Abstract method")

    value = property(_get_value)


class _DictKeyWrap(dict):
    # Wraps a dictionary's key access. For a key of a type `TOrig`, this
    # dictionary will provide a key of type `TProxy`, that should proxy the
    # original key.
    def __init__(self, dict_in, key_wrap, key_unwrap):
        # @param dict_in Dictionary with keys of types TOrig (not necessarily
        # homogeneous).
        # @param key_wrap Functor that maps from TOrig -> TProxy.
        # @param key_unwrap Functor that maps from TProxy -> TOrig.
        dict.__init__(self)
        # N.B. Passing properties to these will cause an issue. This can be
        # sidestepped by storing the properties in a `dict`.
        self._key_wrap = key_wrap
        self._key_unwrap = key_unwrap
        for key, value in dict_in.items():
            self[key] = value

    def __setitem__(self, key, value):
        return dict.__setitem__(self, self._key_wrap(key), value)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._key_wrap(key))

    def __delitem__(self, key):
        return dict.__delitem__(self, self._key_wrap(key))

    def __contains__(self, key):
        return dict.__contains__(self, self._key_wrap(key))

    def items(self):
        return zip(self.keys(), self.values())

    def keys(self):
        return (self._key_unwrap(key) for key in dict.keys(self))

    def raw(self):
        """Returns a dict with the original keys.

        Note:
            Copying to a `dict` will maintain the proxy keys.
        """
        return dict(self.items())


class EqualToDict(_DictKeyWrap):
    """Implements a dictionary where keys are compared using type and
    `lhs.EqualTo(rhs)`.
    """
    def __init__(self, *args, **kwargs):
        class Proxy(_EqualityProxyBase):
            def __eq__(self, other):
                T = type(self.value)
                return (isinstance(other.value, T)
                        and self.value.EqualTo(other.value))

            # https://stackoverflow.com/a/1608907/7829525
            __hash__ = _EqualityProxyBase.__hash__

        dict_in = dict(*args, **kwargs)
        _DictKeyWrap.__init__(self, dict_in, Proxy, Proxy._get_value)


class NamedViewBase:
    """Base for classes generated by ``namedview``.

    Inspired by: https://gitlab.com/ericvsmith/namedlist
    """

    _fields = None  # To be specified by inherited classes.

    def __init__(self, value):
        """Creates a view on ``value``. Any mutations on this instance will be
        reflected in ``value``, and any mutations on ``value`` will be
        reflected in this instance."""
        assert self._fields is not None, (
            "Class must be generated by ``namedview``")
        assert len(self._fields) == len(value)
        object.__setattr__(self, '_value', value)

    @classmethod
    def get_fields(cls):
        """Returns all fields for this class or object."""
        return cls._fields

    def __getitem__(self, i):
        return self._value[i]

    def __setitem__(self, i, value_i):
        self._value[i] = value_i

    def __setattr__(self, name, value):
        """Prevent setting additional attributes."""
        if not hasattr(self, name):
            raise AttributeError(
                "Cannot add attributes! The fields in this named view are"
                f"{self.get_fields()}, but you tried to set '{name}'.")
        object.__setattr__(self, name, value)

    def __len__(self):
        return len(self._value)

    def __iter__(self):
        return iter(self._value)

    def __array__(self):
        """Proxy for use with NumPy."""
        return np.asarray(self._value)

    def _str_like(self, *, per_field_op):
        """Provides human-readable breakout of each field and value."""
        value_strs = []
        for i, field in enumerate(self._fields):
            value_strs.append("{}={}".format(field, per_field_op(self[i])))
        return "{}({})".format(self.__class__.__name__, ", ".join(value_strs))

    def __str__(self):
        return self._str_like(per_field_op=str)

    def __repr__(self):
        inner_text = self._str_like(per_field_op=repr)
        return f"<{inner_text}>"

    @staticmethod
    def _item_property(i):
        # Maps an item (at a given index) to a property.
        return property(fget=lambda self: self[i],
                        fset=lambda self, value: self.__setitem__(i, value))

    @classmethod
    def Zero(cls):
        """Constructs a view onto values set to all zeros."""
        return cls([0] * len(cls._fields))


def _sanitize_field_name(name: str):
    result = name
    # Ensure the first character is a valid opener (e.g., no numbers allowed).
    if not result[0].isidentifier():
        result = "_" + result
    # Ensure that each additional character is valid in turn, avoiding the
    # special case for opening characters by prepending "_" during the check.
    for i in range(1, len(result)):
        if not ("_" + result[i]).isidentifier():
            result = result[:i] + "_" + result[i+1:]
    result = re.sub("__+", "_", result)
    assert result.isidentifier(), f"Sanitization failed on {name} => {result}"
    return result


def namedview(name, fields, *, sanitize_field_names=True):
    """
    Creates a class that is a named view with given ``fields``. When the class
    is instantiated, it must be given the object that it will be a proxy for.
    Similar to ``namedtuple``.

    If ``sanitize_field_names`` is True (the default), then any characters in
    ``fields`` which are not valid in Python identifiers will be automatically
    replaced with `_`. Leading numbers will have `_` inserted, and duplicate
    `_` will be replaced by a single `_`.

    Example:
        ::

            MyView = namedview("MyView", ('a', 'b'))

            value = np.array([1, 2])
            view = MyView(value)
            view.a = 10  # `value` is now [10, 2]
            value[1] = 100  # `view` is now [10, 100]
            view[:] = 3  # `value` is now [3, 3]

            # Get an array from the view *aliasing* the original vector.
            value_view = view[:]
            # Another way to get an aliased array.
            value_view_2 = np.asarray(view)
            # Get an array from the view that is a *copy* of the original
            # vector.
            value_copy = np.array(view)

    Warning:

        As illustrated above, if you use ``np.array(view)``, then it will
        provide a *copied* array from the view. If you want an *aliased* array
        from the view, then use operations like ``view[:]``,
        ``np.asarray(view)``, or ``np.array(view, copy=False)``.

    For more details, see ``NamedViewBase``.
    """
    base_cls = (NamedViewBase, )
    if sanitize_field_names:
        fields = [_sanitize_field_name(f) for f in fields]
    assert len(set(fields)) == len(fields), "Field names must be unique"
    type_dict = dict(_fields=tuple(fields))
    for i, field in enumerate(fields):
        type_dict[field] = NamedViewBase._item_property(i)
    cls = type(name, base_cls, type_dict)
    return cls
