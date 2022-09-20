from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree
from scipy import constants

import ocsmesh.utils as utils

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
        and evaluate whether it satisfies the constraint
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

        return_values = values.copy().ravel()
        bound_values = ref_values.copy().ravel()
        coords = locations.reshape(-1, 2)

        if self._rate is None:
            return values # TODO: COPY?

        if len(coords) != len(return_values):
            raise ValueError(
                "Number of locations and values"
                + f" don't match: {len(coords)} vs {len(return_values)}")

        mask_r = mask.copy().ravel()
        nomask_r = np.logical_not(mask_r)

        points = coords[mask_r]
        xy = coords[nomask_r]

        tree = cKDTree(points)
        near_dists, neighbors = tree.query(xy)
        temp_values = return_values[mask_r][neighbors] * (
                1 + near_dists * self._rate * self.rate_sign)

        # NOTE: No bounds are applied for rate
        mask2 = np.logical_not(self.satisfies(
                     return_values[nomask_r], temp_values))
        # Double indexing copies, we want to modify "return_values"
        temp_values_2 = return_values[nomask_r]
        temp_values_2[mask2] = temp_values[mask2]
        return_values[nomask_r] = temp_values_2

        return_values = return_values.reshape(values.shape)
        return  return_values


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

        return_values = old_values.copy()

        mask = ((ref_values > lower_bound) &
                (ref_values < upper_bound) &
                (np.logical_not(self.satisfies(return_values, self.value))))
        return_values[mask] = self.value

        return_values = self._apply_rate(ref_values, return_values, locations, mask)

        return return_values



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

        return_values = old_values.copy()
        temp_values = self._func(ref_values)

        mask = ((ref_values > lower_bound) &
                (ref_values < upper_bound) &
                (np.logical_not(self.satisfies(return_values, temp_values))))
        return_values[mask] = temp_values[mask]

        return_values = self._apply_rate(ref_values, return_values, locations, mask)

        return return_values


class CourantNumConstraint(Constraint):
    '''Class for defining mesh size constraint based on Courant number

    Methods
    -------
    apply
        Calculate and return he new size function at input reference points
    '''

    def __init__(
            self,
            value: float,
            timestep: float = 150,
            wave_amplitude: float = 2.0,
            value_type: str = 'max',
            ):
        '''Constaint for enforcing bound on Courant number

        Parameters
        ----------
        value : float
            The value of Courant number to enforce
        timestep : float, default=150
            The timestep (in seconds) used to calculate Courant number
        wave_amplitude : float, default=2.0
            Amplitude of wave for linear wave theory approximation
        value_type : {'min', 'max'}, default='min'
            Indicate whether to enforce the input value as min or max of Courant #
        '''

        super().__init__(value_type, rate=None)

        self._value = value
        self._dt = timestep
        self._nu = wave_amplitude


    def apply(
        self,
        ref_values,
        old_values,
        *args,
        **kwargs
        ):
        '''Calculate the new values of size function based on input reference

        Parameters
        ----------
        ref_values : array of floats
            Depth values to be used for Courant number approximations
        old_values : array of floats
            Values of mesh size function before applying the constraint
        \*args : list
            List of arguments not handled by this apply method (
            used in other constraints)
        \*\*kwargs : dict
            Dictionary of arguments not handled by this apply method (
            used in other constraints)
            
        Returns
        -------
        array of floats
            New values of size function after application of the constraint
        '''

        if ref_values.shape != old_values.shape:
            raise ValueError("Shapes of depths and sizes arrays don't match")

        return_values = old_values.copy()

        u_mag = utils.estimate_particle_velocity_from_depth(
            ref_values, self._nu
        )
        depth_mask = utils.can_velocity_be_approximated_by_linear_wave_theory(
            ref_values, self._nu
        )
        char_vel = u_mag + np.sqrt(constants.g * np.abs(ref_values))
        # For overland where h < nu the characteristic velocity is 2 * sqrt(g*h)
        char_vel[~depth_mask] = 2 * u_mag[~depth_mask]
        
        temp_values = utils.get_element_size_courant(
            char_vel, self._dt, self._value
        )
        old_C_apprx = utils.approximate_courant_number_for_depth(
            ref_values, self._dt, old_values, self._nu
        )

        # NOTE: Condition is evaluated on Courant # NOT the element size
        value_mask = np.logical_not(self.satisfies(old_C_apprx, self._value))
        return_values[value_mask] = temp_values[value_mask]

        return return_values
