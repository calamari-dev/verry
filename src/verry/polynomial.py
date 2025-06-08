from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Self

from verry.interval.interval import Interval
from verry.typing import SignedComparable


class IntervalPolynomial[T1: Interval, T2: SignedComparable = Any]:
    __slots__ = ("_dict", "_intvl", "_nvar")
    _dict: defaultdict[tuple[int, ...], T1]
    _intvl: type[T1]
    _nvar: int

    def __init__(
        self, coeffs: Mapping[tuple[int, ...], T1 | T2 | float | int], intvl: type[T1]
    ):
        if not isinstance(intvl, Interval):
            raise TypeError

        if len(coeffs) == 0:
            raise ValueError("coeffs must be non-empty")

        self._dict = defaultdict(intvl)
        self._intvl = intvl
        self._nvar = len(next(iter(coeffs.keys())))

        for key, value in coeffs.items():
            if not isinstance(key, tuple):
                raise TypeError

            if not (len(key) == self._nvar and all(x >= 0 for x in key)):
                raise IndexError

            value = intvl.ensure(value)

            if not value.inf == value.sup == intvl.operator.ZERO:
                self._dict[key] = value

    @property
    def data(self) -> Mapping[tuple[int, ...], T1]:
        return MappingProxyType(self._dict)

    @property
    def nvar(self) -> int:
        return self._nvar

    @property
    def interval(self) -> type[T1]:
        return self._intvl

    @classmethod
    def const(cls, nvar: int, value: T1) -> Self:
        if not isinstance(value, Interval):
            raise TypeError

        return cls({(0,) * nvar: value}, type(value))

    def assign(self, index: int, arg: T1 | T2 | float | int) -> Self:
        deg = max((x[index] for x in self._dict.keys()), default=0)
        coeffs = [self.const(self._nvar, self._intvl()) for _ in range(deg + 1)]

        for key, value in self._dict.items():
            tmp = tuple(0 if i == index else x for i, x in enumerate(key))
            coeffs[key[index]][tmp] = value

        result = coeffs[-1]

        for x in coeffs[:-1]:
            result *= arg
            result += x

        return result

    def copy(self) -> Self:
        if not self._dict:
            return self.__class__({(0,) * self._nvar: 0}, self._intvl)

        return self.__class__(self._dict, self._intvl)

    def deg(self) -> int:
        return max((sum(x) for x in self._dict), default=-1)

    def eval(self, *args: T1 | T2 | float | int) -> T1:
        if len(args) != self._nvar:
            raise ValueError

        if not all(self._intvl.ensurable(x) for x in args):
            raise TypeError

        if not self._dict:
            return self._intvl()

        if self._nvar == 1:
            return self.assign(0, args[0])[(0,)]

        deg = max(x[-1] for x in self._dict.keys())
        coeffs = [self.const(self._nvar - 1, self._intvl()) for _ in range(deg + 1)]

        for key, value in self._dict.items():
            coeffs[key[-1]][key[:-1]] = value

        result = coeffs[-1].copy()

        for x in coeffs[:-1]:
            result *= args[-1]
            result += x

        return result.eval(*args[:-1])

    def reduce(self, index: int) -> Self:
        if self._nvar == 1:
            raise RuntimeError

        if not 0 <= index < self._nvar:
            raise ValueError

        result = self.const(self._nvar - 1, self._intvl())

        for key, value in self._dict.items():
            if key[index] == 0:
                tmp = tuple(x for i, x in enumerate(key) if i != index)
                result[tmp] = value

        return result

    def scale(self, *args: T1 | T2 | int | float) -> Self:
        if len(args) != self._nvar:
            raise ValueError

        if not all(self._intvl.ensurable(x) for x in args):
            raise TypeError

        result = self.copy()

        for key in self._dict:
            for i, x in zip(key, args):
                result[key] *= x**i

        return result

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return NotImplemented

        return other._intvl is self._intvl and other._dict == self._dict

    def __call__(self, *args: T1 | T2 | float | int) -> T1:
        return self.eval(*args)

    def __getitem__(self, key: tuple[int, ...]) -> T1:
        if not (len(key) == self._nvar and all(x >= 0 for x in key)):
            raise KeyError

        return self._dict[key]

    def __setitem__(self, key: tuple[int, ...], value: T1 | T2 | float | int) -> None:
        if not (len(key) == self._nvar and all(x >= 0 for x in key)):
            raise KeyError

        value = self._intvl.ensure(value)

        if not value.inf == value.sup == self._intvl.operator.ZERO:
            self._dict[key] = value
            return

        if key in self._dict:
            del self._dict[key]

    def __add__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        match rhs:
            case self._intvl() | self._intvl.endtype() | float() | int():
                result = self.copy()
                result[(0,) * self._nvar] += rhs
                return result

            case self.__class__():
                result = self.copy()

                for key, value in rhs._dict.items():
                    result[key] += value

                return result

        return NotImplemented

    def __sub__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        match rhs:
            case self._intvl() | self._intvl.endtype() | float() | int():
                result = self.copy()
                result[(0,) * self._nvar] -= rhs
                return result

            case self.__class__():
                result = self.copy()

                for key, value in rhs._dict.items():
                    result[key] -= value

                return result

        return NotImplemented

    def __mul__(self, rhs: Self | T1 | T2 | float | int) -> Self:
        match rhs:
            case self._intvl() | self._intvl.endtype() | float() | int():
                result = self.copy()

                for key in self._dict.keys():
                    result[key] *= rhs

                return result

            case self.__class__():
                result = self.const(self._nvar, self._intvl())

                for lkey, lvalue in self._dict.items():
                    for rkey, rvalue in rhs._dict.items():
                        key = tuple(x + y for x, y in zip(lkey, rkey))
                        result[key] += lvalue * rvalue

                return result

        return NotImplemented

    def __truediv__(self, rhs: T1 | T2 | float | int) -> Self:
        if not isinstance(rhs, (self.interval, self.interval.endtype, float, int)):
            return NotImplemented

        result = self.copy()

        for key in self.data.keys():
            result[key] /= rhs

        return result

    def __pow__(self, rhs: int) -> Self:
        if not (isinstance(rhs, int) and rhs >= 0):
            return NotImplemented

        result = self.const(self._nvar, self._intvl(1))
        tmp = self.copy()

        while rhs != 0:
            if rhs % 2 != 0:
                result *= tmp

            rhs //= 2
            tmp *= tmp

        return result

    def __neg__(self) -> Self:
        if not self._dict:
            return self.const(self._nvar, self._intvl())

        coeffs = {key: -value for key, value in self._dict.items()}
        return self.__class__(coeffs, self._intvl)

    def __pos__(self) -> Self:
        return self.copy()

    def __copy__(self) -> Self:
        return self.copy()
