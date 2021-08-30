from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

ConstraintValueType = Enum("ConstraintValueType", "MIN MAX")

class Constraint(ABC):

    def __init__(self, value_type: str = 'min'):

        if isinstance(value_type, ConstraintValueType):
            self._val_type = value_type
        elif isinstance(value_type, str):
            self._val_type = ConstraintValueType[value_type.upper()]
        else:
            raise ValueError("Invalid input for value type!")


    @property
    def type(self):
        return type(self)


    @property
    def value_type(self):
        return self._val_type

    @property
    def satisfies(self):

        '''
        The function to compare a value with the constraint value
        and evaluate wether it satisfies the constraint
        function's needs to receive values to check as first argument 
        '''

        if self.value_type == ConstraintValueType.MIN:
            return np.greater
        elif self.value_type == ConstraintValueType.MAX:
            return np.less
        else:
            raise ValueError("Invalid value type for constraint!")


    @abstractmethod
    def apply(self):
        pass




class TopoConstConstraint(Constraint):

    def __init__(
            self,
            value,
            upper_bound=np.inf,
            lower_bound=-np.inf,
            value_type: str = 'min'):

        super().__init__(value_type)
        self._lb = lower_bound
        self._ub = upper_bound

        self._val = value


    @property
    def value(self):
        return self._val




    @property
    def topo_bounds(self):

        return self._lb, self._ub


    def apply(self, ref_values, old_values):

        lower_bound, upper_bound = self.topo_bounds

        new_values = old_values.copy()
        new_values[
            (ref_values > lower_bound) &
            (ref_values < upper_bound) &
            (np.logical_not(self.satisfies(new_values, self.value)))
            ] = self.value

        return new_values



class TopoFuncConstraint(Constraint):

    def __init__(
            self,
            function=lambda i: i / 2.0,
            upper_bound=np.inf,
            lower_bound=-np.inf,
            value_type: str = 'min'):

        super().__init__(value_type)

        self._lb = lower_bound
        self._ub = upper_bound

        self._func = lambda i: i / 2.0
        if callable(function):
            self._func = function


    @property
    def topo_bounds(self):

        return self._lb, self._ub


    def apply(self, ref_values, old_values):

        lower_bound, upper_bound = self.topo_bounds

        new_values = old_values.copy()
        temp_values = self._func(ref_values)

        mask = ((ref_values > lower_bound) &
                (ref_values < upper_bound) &
                (np.logical_not(self.satisfies(new_values, temp_values))))
        new_values[mask] = temp_values[mask]

        return new_values
