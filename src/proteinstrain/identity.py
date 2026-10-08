"""Native residue and atom identities shared throughout the package."""

import re
from collections import namedtuple
import numpy as np

_HETERO = re.compile("^(H_\\S+)/(-?\\d+)([A-Za-z]?)$")
_NUMBERED = re.compile("^(-?\\d+)([A-Za-z]?)$")


def space(value):
    """Return a blank-normalised mmCIF field: empty, None and '.' become ' '."""
    text = "" if value is None else str(value)
    return text if text.strip() and text.strip() not in {".", "?"} else " "


class ResidueId(namedtuple("_ResidueId", "hetero number insertion")):
    """Residue identity, normalised on construction and Bio.PDB tuple compatible."""

    __slots__ = ()

    def __new__(cls, hetero=" ", number=0, insertion=" "):
        return super().__new__(cls, space(hetero), int(number), space(insertion))

    @classmethod
    def parse(cls, value):
        """Return a ResidueId from a ResidueId, a 3-tuple, an integer, or '12', '12B', 'H_ATP/501'."""
        if isinstance(value, ResidueId):
            return value
        if isinstance(value, (tuple, list)):
            if len(value) != 3:
                raise ValueError("A residue identity tuple must have three fields.")
            return cls(*value)
        if isinstance(value, (int, np.integer)):
            return cls(" ", int(value), " ")
        text = str(value).strip()
        for pattern, hetero_group in ((_HETERO, 1), (_NUMBERED, None)):
            match = pattern.match(text)
            if match:
                hetero = match.group(hetero_group) if hetero_group else " "
                number = match.group(2 if hetero_group else 1)
                insertion = match.group(3 if hetero_group else 2)
                return cls(hetero, number, insertion)
        raise ValueError(f"Cannot parse residue identity {value!r}.")

    @property
    def is_hetero(self):
        """Whether the residue came from a HETATM record."""
        return self.hetero != " "

    def __str__(self):
        number = f"{self.number}{self.insertion.strip()}"
        return f"{self.hetero}/{number}" if self.is_hetero else number
