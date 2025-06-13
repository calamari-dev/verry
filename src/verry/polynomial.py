import itertools
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self, overload

from verry.interval.interval import Interval
from verry.typing import SignedComparable


def _combination(a: Sequence[int], b: Sequence[int]) -> int:
    result = 1

    for x, y in zip(a, b, strict=True):
        if not x >= y >= 0:
            raise ValueError

        result *= math.factorial(x) // math.factorial(x - y)

    return result


def _prodrange(stop: Sequence[int]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*(range(n) for n in stop))


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

        if next(iter(coeffs), None) is None:
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
    def interval(self) -> type[T1]:
        return self._intvl

    @property
    def ndeg(self) -> tuple[int, ...]:
        return tuple(max(x[i] for x in self._dict) for i in range(self._nvar))

    @property
    def nvar(self) -> int:
        return self._nvar

    @classmethod
    def const(cls, nvar: int, value: T1) -> Self:
        if not isinstance(value, Interval):
            raise TypeError

        if nvar <= 0:
            raise ValueError

        return cls({(0,) * nvar: value}, type(value))

    @classmethod
    def variables(cls, nvar: int, intvl: type[T1]) -> tuple[T1, ...]:
        if not issubclass(intvl, Interval):
            raise TypeError

        if nvar <= 0:
            raise ValueError

        result: list[T1] = []

        for i in range(nvar):
            key = tuple(1 if x == i else 0 for x in range(nvar))
            result.append(cls({key: intvl.operator.ONE}, intvl))

        return tuple(result)

    def assign(self, index: int, arg: T1 | T2 | float | int) -> Self:
        if not 0 <= index < self._nvar:
            raise ValueError

        deg = max((x[index] for x in self._dict.keys()), default=0)
        coeffs = [self.const(self._nvar, self._intvl()) for _ in range(deg + 1)]

        for key, value in self._dict.items():
            tmp = tuple(key[i] if i != index else 0 for i in range(self._nvar))
            coeffs[key[index]][tmp] = value

        result = coeffs[-1]

        for x in reversed(coeffs[:-1]):
            result *= arg
            result += x

        return result

    def compose(self, index: int, arg: Self) -> Self:
        if not 0 <= index < self._nvar:
            raise ValueError

        deg = max((x[index] for x in self._dict.keys()), default=0)
        coeffs = [self.const(self._nvar, self._intvl()) for _ in range(deg + 1)]

        for key, value in self._dict.items():
            tmp = tuple(key[i] if i != index else 0 for i in range(self._nvar))
            coeffs[key[index]][tmp] = value

        result = coeffs[-1]

        for x in reversed(coeffs[:-1]):
            result *= arg
            result += x

        return result

    def copy(self) -> Self:
        if not self._dict:
            key = (0,) * self._nvar
            return self.__class__({key: self._intvl.operator.ZERO}, self._intvl)

        return self.__class__(self._dict, self._intvl)

    def deg(self) -> int:
        return max((sum(x) for x in self._dict), default=-1)

    def eval(self, *args: T1 | T2 | float | int) -> T1:
        if len(args) != self._nvar:
            raise ValueError

        if not isinstance(args[-1], (self._intvl, self._intvl.endtype, float, int)):
            raise TypeError

        if not self._dict:
            return self._intvl()

        if self._nvar == 1:
            return self.assign(0, args[0])[(0,)]

        deg = max(x[-1] for x in self._dict.keys())
        coeffs = [self.const(self._nvar - 1, self._intvl()) for _ in range(deg + 1)]

        for key, value in self._dict.items():
            coeffs[key[-1]]._dict[key[:-1]] = value

        result = coeffs[-1].copy()

        for x in reversed(coeffs[:-1]):
            result *= args[-1]
            result += x

        return result.eval(*args[:-1])

    def range(self, *args: T1) -> T1:
        var = self.variables(self._nvar, self._intvl)
        normalized = self.copy()

        for i, arg in enumerate(args):
            tmp = (var[i] - arg.inf) / (self._intvl(arg.sup) - arg.inf)
            normalized = normalized.compose(i, tmp)

        bcoeffs: dict[tuple[int, ...], T1] = {}
        ndeg = self.ndeg

        for i in _prodrange(ndeg):
            bcoeffs[i] = normalized[i] / _combination(ndeg, i)

        for r in range(self._nvar):
            for k in range(1, ndeg[r] + 1):
                tmp: dict[tuple[int, ...], T1] = {}

                for i in _prodrange(ndeg):
                    if i[r] < k:
                        tmp[i] = bcoeffs[i]
                    else:
                        j = tuple(x - 1 if x == r else x for x in range(self._nvar))
                        tmp[i] = bcoeffs[i] + bcoeffs[j]

                bcoeffs = tmp

        return self._intvl.hull(*bcoeffs.values())

    def reduce(self, index: int) -> Self:
        if self._nvar == 1:
            raise RuntimeError

        if not 0 <= index < self._nvar:
            raise ValueError

        result = self.const(self._nvar - 1, self._intvl())

        for key, value in filter(lambda x, _: x[index] == 0, self._dict.items()):
            tmp = tuple(key[i] for i in range(self._nvar) if i != index)
            result._dict[tmp] = value

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
        if not isinstance(rhs, (self._intvl, self._intvl.endtype, float, int)):
            return NotImplemented

        result = self.copy()

        for key in self.data.keys():
            result[key] /= rhs

        return result

    def __pow__(self, rhs: int) -> Self:
        if not (isinstance(rhs, int) and rhs >= 0):
            return NotImplemented

        ONE = self._intvl.operator.ONE
        result = self.const(self._nvar, self._intvl(ONE))
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
