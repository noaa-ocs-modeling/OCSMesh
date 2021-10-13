from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree

ConstraintValueType = Enum("ConstraintValueType", "MIN MAX")

class Constraint(ABC):

    def __init__(self, value_type: str = 'min', rate: float = 0.1):

        # TODO: Put rate in a mixin ?
        self._rate = rate

        # pylint: disable=W1116
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

        # pylint: disable=R1705
        if self.value_type == ConstraintValueType.MIN:
            return np.greater
        elif self.value_type == ConstraintValueType.MAX:
            return np.less

        raise ValueError("Invalid value type for constraint!")

    @property
    def rate_sign(self):
        # TODO: Put this method in a mixin

        '''
        Based on the value-type of the constraints, return a sign
        indicating whether rate is for expansion or contraction of
        size outside calculated zone
        '''

        # pylint: disable=R1705
        if self.value_type == ConstraintValueType.MIN:
            return -1
        elif self.value_type == ConstraintValueType.MAX:
            return +1

        raise ValueError("Invalid value type for constraint!")


    @abstractmethod
    def apply(self):
        pass


    def _apply_rate(self, ref_values, values, locations, mask):

        if not np.any(mask):
            return values # TODO: COPY?

        new_values = values.copy().ravel()
        bound_values = ref_values.copy().ravel()
        coords = locations.reshape(-1, 2)

        if self._rate is None:
            return values # TODO: COPY?

        if len(coords) != len(new_values):
            raise ValueError(
                "Number of locations and values"
                + f" don't match: {len(coords)} vs {len(new_values)}")

        mask_r = mask.copy().ravel()
        nomask_r = np.logical_not(mask_r)

        points = coords[mask_r]
        xy = coords[nomask_r]

        tree = cKDTree(points)
        near_dists, neighbors = tree.query(xy)
        temp_values = new_values[mask_r][neighbors] * (
                1 + near_dists * self._rate * self.rate_sign)

        # NOTE: No bounds are applied for rate
        mask2 = np.logical_not(self.satisfies(
                     new_values[nomask_r], temp_values))
        # Double indexing copies, we want to modify "new_values"
        temp_values_2 = new_values[nomask_r]
        temp_values_2[mask2] = temp_values[mask2]
        new_values[nomask_r] = temp_values_2

        new_values = new_values.reshape(values.shape)
        return  new_values


# TODO:
class BoundMixin:
    pass


class RateMixin:
    pass




class TopoConstConstraint(Constraint):

    def __init__(
            self,
            value,
            upper_bound=np.inf,
            lower_bound=-np.inf,
            value_type: str = 'min',
            rate=None):

        super().__init__(value_type, rate)

        self._lb = lower_bound
        self._ub = upper_bound

        self._val = value


    @property
    def value(self):
        return self._val




    @property
    def topo_bounds(self):

        return self._lb, self._ub


    def apply(self, ref_values, old_values, locations=None):

        lower_bound, upper_bound = self.topo_bounds

        new_values = old_values.copy()

        mask = ((ref_values > lower_bound) &
                (ref_values < upper_bound) &
                (np.logical_not(self.satisfies(new_values, self.value))))
        new_values[mask] = self.value

        new_values = self._apply_rate(ref_values, new_values, locations, mask)

        return new_values



class TopoFuncConstraint(Constraint):

    def __init__(
            self,
            function=lambda i: i / 2.0,
            upper_bound=np.inf,
            lower_bound=-np.inf,
            value_type: str = 'min',
            rate=None):

        super().__init__(value_type, rate)

        self._lb = lower_bound
        self._ub = upper_bound

        self._func = lambda i: i / 2.0
        if callable(function):
            self._func = function


    @property
    def topo_bounds(self):

        return self._lb, self._ub


    def apply(self, ref_values, old_values, locations=None):

        lower_bound, upper_bound = self.topo_bounds

        new_values = old_values.copy()
        temp_values = self._func(ref_values)

        mask = ((ref_values > lower_bound) &
                (ref_values < upper_bound) &
                (np.logical_not(self.satisfies(new_values, temp_values))))
        new_values[mask] = temp_values[mask]

        new_values = self._apply_rate(ref_values, new_values, locations, mask)

        return new_values
