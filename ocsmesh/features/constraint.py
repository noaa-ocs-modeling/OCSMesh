from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

ConstraintValueType = Enum("ConstraintValueType", "MIN MAX")

class Constraint(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @property
    def type(self):
        return type(self)

    @property
    def value_type(self):
        return self._val_type

    @property
    def value(self):
        return self._val

    @property
    def check(self):

        '''
        The function to compare a value with the constraint value
        and evaluate wether it satisfies the constraint
        '''

        if self.value_type == ConstraintValueType.MIN:
            return lambda arr: arr > self.value
        elif self.value_type == ConstraintValueType.MAX:
            return lambda arr: arr < self.value


class TopoConstraint(Constraint):

    def __init__(
            self,
            value,
            upper_bound=np.inf,
            lower_bound=-np.inf,
            value_type: str = 'min'):

        self._lb = lower_bound
        self._ub = upper_bound

        if isinstance(value_type, ConstraintValueType):
            self._val_type = value_type
        elif isinstance(value_type, str):
            self._val_type = ConstraintValueType[value_type.upper()]
        else:
            raise ValueError("Invalid input for value type!")
        self._val = value

    @property
    def topo_bounds(self):

        return self._lb, self._ub
