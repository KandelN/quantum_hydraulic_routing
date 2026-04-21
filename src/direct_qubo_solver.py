from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp

from direct_qubo_system import DirectQUBOConfig, build_direct_nonlinear_objective


@dataclass
class DirectQUBOResult:
    Q: np.ndarray
    y: np.ndarray
    objective_value: float
    num_binary_variables: int
    variables: list[str]
    reduced_to_qubo: bool = False
    num_auxiliary_variables: int = 0


@dataclass
class SymbolEncodingSpec:
    symbol: sp.Symbol
    label: str
    bits: int
    scale: float


@dataclass
class SymbolicBinarySolveResult:
    encoded_values: dict[str, float]
    objective_value: float
    num_binary_variables: int
    variables: list[str]
    reduced_to_qubo: bool = False
    num_auxiliary_variables: int = 0


class DirectQUBOBuilder:
    def __init__(self, cfg: DirectQUBOConfig):
        self.cfg = cfg
        self.bits: dict[tuple[str, int, int], sp.Symbol] = {}

    def z(self, name: str, i: int, bit: int):
        key = (name, i, bit)
        if key not in self.bits:
            self.bits[key] = sp.Symbol(f'z_{name}_{i}_{bit}', binary=True)
        return self.bits[key]

    def encode(self, name: str, i: int, m: int, s: float):
        scale = sp.nsimplify(s)
        return sum(scale * sp.Rational(1, 2**bit) * self.z(name, i, bit) for bit in range(1, m + 1))


class SymbolicBinaryBuilder:
    def __init__(self):
        self.bits: dict[tuple[str, int], sp.Symbol] = {}

    def z(self, label: str, bit: int) -> sp.Symbol:
        key = (label, bit)
        if key not in self.bits:
            self.bits[key] = sp.Symbol(f'z_{label}_{bit}', binary=True)
        return self.bits[key]

    def encode(self, label: str, bits: int, scale: float) -> sp.Expr:
        scale_sym = sp.nsimplify(scale)
        return sum(scale_sym * sp.Rational(1, 2**bit) * self.z(label, bit) for bit in range(1, bits + 1))


class RosenbergReducer:
    """Reduce a polynomial pseudo-Boolean objective to quadratic form.

    This applies repeated pair substitution with the standard Rosenberg
    penalty M(xy - 2xw - 2yw + 3w), where w is a fresh auxiliary binary
    variable. The input expression must already be polynomial in binary
    variables.
    """

    def __init__(self, penalty: float = 100.0):
        self.penalty = float(penalty)
        self.aux_vars: list[sp.Symbol] = []

    def new_aux(self) -> sp.Symbol:
        aux = sp.Symbol(f'aux_{len(self.aux_vars) + 1}', binary=True)
        self.aux_vars.append(aux)
        return aux

    @staticmethod
    def _binary_monomial(term: sp.Expr) -> tuple[sp.Expr, list[sp.Symbol]]:
        coeff, monomial = term.as_coeff_Mul()
        vars_: list[sp.Symbol] = []
        for factor, exp in monomial.as_powers_dict().items():
            if getattr(factor, 'is_Symbol', False):
                if int(exp) >= 1:
                    vars_.append(factor)
            else:
                coeff *= factor**exp
        vars_.sort(key=lambda s: s.name)
        return sp.expand(coeff), vars_

    @staticmethod
    def multilinearize(expr: sp.Expr) -> sp.Expr:
        expr = sp.expand(expr)
        new_terms: list[sp.Expr] = []
        for term in sp.Add.make_args(expr):
            coeff, vars_ = RosenbergReducer._binary_monomial(term)
            prod = sp.Integer(1)
            for var in vars_:
                prod *= var
            new_terms.append(sp.expand(coeff * prod))
        return sp.expand(sum(new_terms))

    @staticmethod
    def is_polynomial(expr: sp.Expr) -> bool:
        try:
            symbols = sorted(expr.free_symbols, key=lambda s: s.name)
            if not symbols:
                return True
            sp.Poly(sp.expand(expr), *symbols)
            return True
        except sp.PolynomialError:
            return False

    @staticmethod
    def total_degree(expr: sp.Expr) -> int:
        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        if not symbols:
            return 0
        return sp.Poly(sp.expand(expr), *symbols).total_degree()

    def reduce(self, expr: sp.Expr) -> sp.Expr:
        if not self.is_polynomial(expr):
            raise ValueError(
                'Rosenberg reduction requires a polynomial pseudo-Boolean objective. '
                'This expression still contains denominators or non-polynomial terms.'
            )

        expr = self.multilinearize(expr)
        changed = True

        while changed:
            changed = False
            new_terms: list[sp.Expr] = []

            for term in sp.Add.make_args(sp.expand(expr)):
                coeff, vars_ = self._binary_monomial(term)

                if len(vars_) <= 2:
                    prod = sp.Integer(1)
                    for var in vars_:
                        prod *= var
                    new_terms.append(sp.expand(coeff * prod))
                    continue

                x, y = vars_[0], vars_[1]
                w = self.new_aux()

                rest = sp.Integer(1)
                for var in vars_[2:]:
                    rest *= var

                new_terms.append(sp.expand(coeff * w * rest))
                new_terms.append(
                    sp.expand(self.penalty * (x * y - 2 * x * w - 2 * y * w + 3 * w))
                )
                changed = True

            expr = self.multilinearize(sum(new_terms))

        return sp.expand(expr)


def _bit_batches(nvars: int, batch_size: int = 8192):
    total = 1 << nvars
    shifts = np.arange(nvars - 1, -1, -1, dtype=np.uint64)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ints = np.arange(start, stop, dtype=np.uint64)
        yield ((ints[:, None] >> shifts) & 1).astype(np.int8)


def brute_force_binary_objective(expr, variables=None, batch_size: int = 8192):
    if variables is None:
        variables = sorted(expr.free_symbols, key=lambda s: s.name)

    func = sp.lambdify(variables, expr, modules='numpy')
    best_bits = None
    best_value = np.inf

    for batch in _bit_batches(len(variables), batch_size=batch_size):
        batch_float = batch.astype(np.float64, copy=False)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            values = np.asarray(func(*batch_float.T), dtype=float).reshape(-1)
        values[~np.isfinite(values)] = np.inf
        idx = int(np.argmin(values))
        val = float(values[idx])
        if np.isfinite(val) and val < best_value:
            best_value = val
            best_bits = batch[idx].copy()

    if best_bits is None:
        raise RuntimeError('Direct QUBO brute-force search found no finite objective value. Reduce bit ranges or scales.')

    return variables, best_bits, best_value


def decode_solution(bits, variables, cfg: DirectQUBOConfig, N: int, Qin_np1=None, y0=None):
    bit_map = {v.name: int(b) for v, b in zip(variables, bits)}

    Q = []
    y = []
    for i in range(N):
        if i == 0 and Qin_np1 is not None:
            q_val = float(Qin_np1)
        else:
            q_val = 0.0
            for bit in range(1, cfg.mQ + 1):
                q_val += cfg.sQ * (2.0 ** (-bit)) * bit_map.get(f'z_Q_{i}_{bit}', 0)

        if i == 0 and y0 is not None:
            y_val = float(y0)
        else:
            y_val = 0.0
            for bit in range(1, cfg.my + 1):
                y_val += cfg.sy * (2.0 ** (-bit)) * bit_map.get(f'z_y_{i}_{bit}', 0)

        Q.append(q_val)
        y.append(y_val)

    return np.asarray(Q, dtype=float), np.asarray(y, dtype=float)


def decode_symbolic_solution(bits, variables, specs: list[SymbolEncodingSpec]) -> dict[str, float]:
    bit_map = {v.name: int(b) for v, b in zip(variables, bits)}
    decoded: dict[str, float] = {}

    for spec in specs:
        value = 0.0
        for bit in range(1, spec.bits + 1):
            value += spec.scale * (2.0 ** (-bit)) * bit_map.get(f'z_{spec.label}_{bit}', 0)
        decoded[str(spec.symbol)] = value

    return decoded


def reduce_to_qubo(expr: sp.Expr, penalty: float = 100.0):
    reducer = RosenbergReducer(penalty=penalty)
    reduced_expr = reducer.reduce(expr)
    return reduced_expr, reducer.aux_vars


def build_binary_objective_from_symbolic_system(
    equations: list[sp.Expr],
    specs: list[SymbolEncodingSpec],
):
    """Start from a real nonlinear symbolic system and build the binary objective.

    Workflow:
      1. user provides nonlinear equations in real variables,
      2. each real variable is replaced by its binary encoding,
      3. the least-squares objective sum(F_k^2) is formed,
      4. the result is multilinearized using x^p = x for binary x.

    This is the generic standalone demo path, independent of direct_qubo_system.py.
    """

    builder = SymbolicBinaryBuilder()
    substitution_map: dict[sp.Symbol, sp.Expr] = {}

    for spec in specs:
        substitution_map[spec.symbol] = builder.encode(spec.label, spec.bits, spec.scale)

    encoded_equations: list[sp.Expr] = []
    for eq in equations:
        encoded = sp.expand(sp.cancel(eq.subs(substitution_map)))
        encoded_equations.append(encoded)

    objective = sp.expand(sum(eq**2 for eq in encoded_equations))
    objective = RosenbergReducer.multilinearize(objective)

    return objective, encoded_equations, substitution_map


def solve_symbolic_nonlinear_binary_system(
    equations: list[sp.Expr],
    specs: list[SymbolEncodingSpec],
    *,
    reduce_objective_to_qubo: bool = False,
    rosenberg_penalty: float = 100.0,
    verbose: bool = True,
):
    objective, encoded_equations, substitution_map = build_binary_objective_from_symbolic_system(
        equations,
        specs,
    )

    solved_expr = objective
    aux_vars: list[sp.Symbol] = []
    if reduce_objective_to_qubo:
        solved_expr, aux_vars = reduce_to_qubo(solved_expr, penalty=rosenberg_penalty)

    variables = sorted(solved_expr.free_symbols, key=lambda s: s.name)
    variables, bits, best_value = brute_force_binary_objective(solved_expr, variables=variables)
    decoded = decode_symbolic_solution(bits, variables, specs)

    if verbose:
        print('Standalone symbolic nonlinear -> binary demo')
        print('Original equations:')
        for i, eq in enumerate(equations, start=1):
            print(f'  f{i} = {sp.expand(eq)}')
        print('Binary encodings:')
        for sym, enc in substitution_map.items():
            print(f'  {sym} = {enc}')
        print('Encoded equations:')
        for i, eq in enumerate(encoded_equations, start=1):
            print(f'  f{i}(z) = {eq}')
        print('Objective degree before reduction:', RosenbergReducer.total_degree(objective))
        if reduce_objective_to_qubo:
            print('Rosenberg reduction enabled')
            print('Reduced degree:', RosenbergReducer.total_degree(solved_expr))
            print('Auxiliary variables:', len(aux_vars), [v.name for v in aux_vars])
        print('Best objective value:', best_value)
        print('Decoded variables:', {k: round(v, 6) for k, v in decoded.items()})

    return SymbolicBinarySolveResult(
        encoded_values=decoded,
        objective_value=float(best_value),
        num_binary_variables=len(variables),
        variables=[v.name for v in variables],
        reduced_to_qubo=reduce_objective_to_qubo,
        num_auxiliary_variables=len(aux_vars),
    )


def solve_direct_qubo_step(
    x: np.ndarray,
    z: np.ndarray,
    Qn: np.ndarray,
    yn: np.ndarray,
    Qin_np1: float,
    dt: float,
    *,
    cfg: DirectQUBOConfig,
    use_friction: bool = True,
    reduce_objective_to_qubo: bool = False,
    rosenberg_penalty: float = 100.0,
    verbose: bool = True,
):
    builder = DirectQUBOBuilder(cfg)
    obj, _, _ = build_direct_nonlinear_objective(
        x,
        z,
        Qn,
        yn,
        Qin_np1,
        dt,
        cfg,
        builder.encode,
        use_friction=use_friction,
    )

    aux_vars: list[sp.Symbol] = []
    solved_expr = obj
    if reduce_objective_to_qubo:
        solved_expr, aux_vars = reduce_to_qubo(obj, penalty=rosenberg_penalty)

    variables = sorted(solved_expr.free_symbols, key=lambda s: s.name)

    if verbose:
        print('Direct nonlinear binary formulation')
        print('Original variable count:', len(sorted(obj.free_symbols, key=lambda s: s.name)))
        print('Original total degree:', RosenbergReducer.total_degree(obj))
        if reduce_objective_to_qubo:
            print('Rosenberg reduction enabled')
            print('Auxiliary variables:', len(aux_vars))
            print('Reduced total degree:', RosenbergReducer.total_degree(solved_expr))
        print('Binary variables:', len(variables))
        print('Variables:', [v.name for v in variables])

    variables, bits, best_value = brute_force_binary_objective(solved_expr, variables=variables)
    Q_new, y_new = decode_solution(bits, variables, cfg, len(x), Qin_np1=Qin_np1, y0=yn[0])

    if verbose:
        print('Best objective value:', best_value)
        print('Decoded Q:', np.round(Q_new, 6))
        print('Decoded y:', np.round(y_new, 6))

    return DirectQUBOResult(
        Q=Q_new,
        y=y_new,
        objective_value=float(best_value),
        num_binary_variables=len(variables),
        variables=[v.name for v in variables],
        reduced_to_qubo=reduce_objective_to_qubo,
        num_auxiliary_variables=len(aux_vars),
    )


def _demo_symbolic_nonlinear_system():
    """Demo requested by the user: start from real nonlinear equations first."""
    q, y = sp.symbols('q y', real=True)

    equations = [
        q * y - 5,
        q + y - 12,
    ]

    specs = [
        SymbolEncodingSpec(symbol=q, label='q', bits=2, scale=4.0),
        SymbolEncodingSpec(symbol=y, label='y', bits=3, scale=4.0),
    ]

    print('=== Demo: real nonlinear system -> binary encoding -> QUBO ===')
    solve_symbolic_nonlinear_binary_system(
        equations,
        specs,
        reduce_objective_to_qubo=False,
        rosenberg_penalty=20.0,
        verbose=True,
    )
    print()


def _demo_direct_swe_small():
    """Small SWE direct demo for one reach / one step using the repo formulation."""
    cfg = DirectQUBOConfig(mQ=2, my=2, sQ=20.0, sy=2.5)
    x = np.array([0.0, 1600.0], dtype=float)
    z = np.array([0.90, 0.5], dtype=float)
    Qn = np.array([20.0, 20.0], dtype=float)
    yn = np.array([1.77, 1.77], dtype=float)
    Qin_np1 = 25.0
    dt = 600.0

    print('=== Demo: small direct SWE binary system ===')
    solve_direct_qubo_step(
        x,
        z,
        Qn,
        yn,
        Qin_np1,
        dt,
        cfg=cfg,
        use_friction=True,
        reduce_objective_to_qubo=True,
        rosenberg_penalty=20.0,
        verbose=True,
    )
    print()


if __name__ == '__main__':
    # _demo_symbolic_nonlinear_system()
    _demo_direct_swe_small()
