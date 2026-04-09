from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp


@dataclass
class QUBOConfig:
    g: float = 9.81
    s0: float = 1.0e-3
    theta: float = 0.5
    dx: float = 100.0
    dt: float = 100.0
    qubo_m: int = 2
    direct_sh: float = 3.5
    direct_su: float = 3.5
    penalty: float = 10.0

    @property
    def scale_h(self) -> float:
        return self.direct_sh / (1.0 - 2.0 ** (-self.qubo_m))

    @property
    def scale_u(self) -> float:
        return self.direct_su / (1.0 - 2.0 ** (-self.qubo_m))


def load_grid_csv(path: Path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))
    x = np.array([float(v) for v in rows[0][1:]], dtype=float)
    t = np.array([float(r[0]) for r in rows[1:]], dtype=float)
    data = []
    for r in rows[1:]:
        row = []
        for v in r[1:]:
            v = v.strip()
            row.append(None if v in {'', '?'} else float(v))
        data.append(row)
    return x, t, np.array(data, dtype=object)


def save_grid_csv(path: Path, x: np.ndarray, t: np.ndarray, data: np.ndarray) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t\\x', *[f'{xi:g}' for xi in x]])
        for ti, row in zip(t, data):
            writer.writerow([f'{ti:g}', *[f'{float(v):.6f}' for v in row]])


def qubo_bruteforce(Q):
    n = Q.shape[0]
    best_q = None
    best_val = np.inf
    for i in range(2 ** n):
        q = np.array(list(np.binary_repr(i, width=n))).astype(int)
        val = q @ Q @ q
        if val < best_val:
            best_val = val
            best_q = q
    return best_q


class RosenbergReducer:
    def __init__(self, penalty: float):
        self.penalty = float(penalty)
        self.aux_vars: list[sp.Symbol] = []
        self.pair_to_aux: dict[frozenset[sp.Symbol], sp.Symbol] = {}

    @staticmethod
    def monomial(term: sp.Expr):
        coeff, mon = term.as_coeff_Mul()
        vars_: list[sp.Symbol] = []
        for factor, exp in mon.as_powers_dict().items():
            if getattr(factor, 'is_Symbol', False):
                if int(exp) >= 1:
                    vars_.append(factor)
            else:
                coeff *= factor ** exp
        vars_.sort(key=lambda s: s.name)
        return sp.expand(coeff), vars_

    @staticmethod
    def multilinearize(expr: sp.Expr) -> sp.Expr:
        expr = sp.expand(expr)
        terms = []
        for term in sp.Add.make_args(expr):
            coeff, vars_ = RosenbergReducer.monomial(term)
            prod = sp.Integer(1)
            for var in vars_:
                prod *= var
            terms.append(sp.expand(coeff * prod))
        return sp.expand(sum(terms))

    def get_aux(self, x: sp.Symbol, y: sp.Symbol) -> tuple[sp.Symbol, bool]:
        key = frozenset((x, y))
        if key in self.pair_to_aux:
            return self.pair_to_aux[key], False
        aux = sp.Symbol(f'aux_{len(self.aux_vars) + 1}', binary=True)
        self.aux_vars.append(aux)
        self.pair_to_aux[key] = aux
        return aux, True

    def reduce(self, expr: sp.Expr) -> sp.Expr:
        expr = self.multilinearize(expr)
        changed = True
        while changed:
            changed = False
            new_terms: list[sp.Expr] = []
            for term in sp.Add.make_args(sp.expand(expr)):
                coeff, vars_ = self.monomial(term)
                if len(vars_) <= 2:
                    prod = sp.Integer(1)
                    for var in vars_:
                        prod *= var
                    new_terms.append(sp.expand(coeff * prod))
                    continue
                x, y = vars_[0], vars_[1]
                aux, created = self.get_aux(x, y)
                rest = sp.Integer(1)
                for var in vars_[2:]:
                    rest *= var
                new_terms.append(sp.expand(coeff * aux * rest))
                if created:
                    M = sp.Float(self.penalty)
                    new_terms.append(sp.expand(M * (x * y - 2 * x * aux - 2 * y * aux + 3 * aux)))
                changed = True
            expr = self.multilinearize(sum(new_terms))
        return sp.expand(expr)


def encode_variable(kind: str, ti: int, xi: int, cfg: QUBOConfig, bit_symbols: dict[str, sp.Symbol]):
    scale = cfg.scale_h if kind == 'h' else cfg.scale_u
    expr = sp.Integer(0)
    names = []
    for bit in range(1, cfg.qubo_m + 1):
        name = f'z_{kind}_{ti}_{xi}_{bit}'
        z = sp.Symbol(name, binary=True)
        bit_symbols[name] = z
        expr += sp.Float(scale) * sp.Rational(1, 2**bit) * z
        names.append(name)
    return sp.expand(expr), names


def build_state_expressions(h_grid, u_grid, cfg: QUBOConfig):
    nt, nx = h_grid.shape
    h_expr = [[None] * nx for _ in range(nt)]
    u_expr = [[None] * nx for _ in range(nt)]
    bit_symbols: dict[str, sp.Symbol] = {}
    unknown_map: dict[tuple[str, int, int], list[str]] = {}

    for ti in range(nt):
        for xi in range(nx):
            hv = h_grid[ti, xi]
            uv = u_grid[ti, xi]
            if hv is None:
                h_expr[ti][xi], bits = encode_variable('h', ti, xi, cfg, bit_symbols)
                unknown_map[('h', ti, xi)] = bits
            else:
                h_expr[ti][xi] = sp.Float(float(hv))
            if uv is None:
                u_expr[ti][xi], bits = encode_variable('u', ti, xi, cfg, bit_symbols)
                unknown_map[('u', ti, xi)] = bits
            else:
                u_expr[ti][xi] = sp.Float(float(uv))
    return h_expr, u_expr, bit_symbols, unknown_map


def continuity_residual(h_expr, u_expr, ti: int, xi: int, cfg: QUBOConfig):
    dx = sp.Float(cfg.dx)
    dt = sp.Float(cfg.dt)
    theta = sp.Float(cfg.theta)
    hi_k = h_expr[ti][xi]
    hj_k = h_expr[ti][xi + 1]
    hi_n = h_expr[ti + 1][xi]
    hj_n = h_expr[ti + 1][xi + 1]
    ui_k = u_expr[ti][xi]
    uj_k = u_expr[ti][xi + 1]
    ui_n = u_expr[ti + 1][xi]
    uj_n = u_expr[ti + 1][xi + 1]
    res = -dx * hi_k - dx * hj_k + dx * hi_n + dx * hj_n
    res += -2 * dt * (1 - theta) * hi_k * ui_k + 2 * dt * (1 - theta) * hj_k * uj_k
    res += -2 * dt * theta * hi_n * ui_n + 2 * dt * theta * hj_n * uj_n
    return sp.expand(res)


def momentum_residual(h_expr, u_expr, ti: int, xi: int, cfg: QUBOConfig):
    dx = sp.Float(cfg.dx)
    dt = sp.Float(cfg.dt)
    theta = sp.Float(cfg.theta)
    g = sp.Float(cfg.g)
    s0 = sp.Float(cfg.s0)
    hi_k = h_expr[ti][xi]
    hj_k = h_expr[ti][xi + 1]
    hi_n = h_expr[ti + 1][xi]
    hj_n = h_expr[ti + 1][xi + 1]
    ui_k = u_expr[ti][xi]
    uj_k = u_expr[ti][xi + 1]
    ui_n = u_expr[ti + 1][xi]
    uj_n = u_expr[ti + 1][xi + 1]
    res = -dx * hi_k * ui_k - dx * hj_k * uj_k + dx * hi_n * ui_n + dx * hj_n * uj_n
    res += -2 * (1 - theta) * dt * hi_k * ui_k**2 + 2 * (1 - theta) * dt * hj_k * uj_k**2
    res += -2 * theta * dt * hi_n * ui_n**2 + 2 * theta * dt * hj_n * uj_n**2
    res += -g * (1 - theta) * dt * hi_k**2 + g * (1 - theta) * dt * hj_k**2
    res += -g * theta * dt * hi_n**2 + g * theta * dt * hj_n**2
    res += -g * dt * dx * s0 * (1 - theta) * hi_k / 2
    res += -g * dt * dx * s0 * (1 - theta) * hj_k / 2
    res += -g * dt * dx * s0 * theta * hi_n / 2
    res += -g * dt * dx * s0 * theta * hj_n / 2
    return sp.expand(res)


def objective_from_grids(h_grid, u_grid, cfg: QUBOConfig):
    h_expr, u_expr, bit_symbols, unknown_map = build_state_expressions(h_grid, u_grid, cfg)
    nt, nx = h_grid.shape
    objective = sp.Integer(0)
    for ti in range(nt - 1):
        for xi in range(nx - 1):
            c = continuity_residual(h_expr, u_expr, ti, xi, cfg)
            m = momentum_residual(h_expr, u_expr, ti, xi, cfg)
            objective += sp.expand(c**2 + m**2)
    objective = RosenbergReducer.multilinearize(objective)
    return sp.expand(objective), bit_symbols, unknown_map


def qubo_matrix_from_expr(expr: sp.Expr):
    expr = sp.expand(expr)
    variables = sorted(expr.free_symbols, key=lambda s: s.name)
    n = len(variables)
    index = {v: i for i, v in enumerate(variables)}
    Q = np.zeros((n, n), dtype=float)
    constant = 0.0
    for term in sp.Add.make_args(expr):
        coeff, vars_ = RosenbergReducer.monomial(term)
        c = float(coeff)
        if len(vars_) == 0:
            constant += c
        elif len(vars_) == 1:
            i = index[vars_[0]]
            Q[i, i] += c
        elif len(vars_) == 2:
            i = index[vars_[0]]
            j = index[vars_[1]]
            if i == j:
                Q[i, i] += c
            else:
                Q[i, j] += 0.5 * c
                Q[j, i] += 0.5 * c
        else:
            raise ValueError('Expression is not quadratic.')
    return Q, variables, constant


def decode_unknowns(best_bits, variables, unknown_map, cfg: QUBOConfig):
    bit_values = {v.name: int(b) for v, b in zip(variables, best_bits)}
    decoded: dict[tuple[str, int, int], float] = {}
    for key, bit_names in unknown_map.items():
        kind = key[0]
        scale = cfg.scale_h if kind == 'h' else cfg.scale_u
        value = 0.0
        for bit, name in enumerate(bit_names, start=1):
            value += scale * (2.0 ** (-bit)) * bit_values.get(name, 0)
        decoded[key] = value
    return decoded


def fill_decoded_grid(grid, decoded, kind: str):
    out = np.array(grid, dtype=object)
    nt, nx = out.shape
    out_num = np.zeros((nt, nx), dtype=float)
    for ti in range(nt):
        for xi in range(nx):
            if out[ti, xi] is None:
                out_num[ti, xi] = decoded[(kind, ti, xi)]
            else:
                out_num[ti, xi] = float(out[ti, xi])
    return out_num


def build_and_optionally_solve(cfg: QUBOConfig, data_dir: Path, output_dir: Path, solve: bool = True, verbose: bool = True):
    h_file = data_dir / 'data_h.csv'
    u_file = data_dir / 'data_u.csv'
    x_h, t_h, h_grid = load_grid_csv(h_file)
    x_u, t_u, u_grid = load_grid_csv(u_file)
    if not np.allclose(x_h, x_u) or not np.allclose(t_h, t_u):
        raise ValueError('h and u CSV grids must have matching x and t coordinates.')

    objective, bit_symbols, unknown_map = objective_from_grids(h_grid, u_grid, cfg)
    reducer = RosenbergReducer(cfg.penalty)
    reduced = reducer.reduce(objective)
    Q, variables, constant = qubo_matrix_from_expr(reduced)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / 'qubo_matrix.csv', Q, delimiter=',')
    np.save(output_dir / 'qubo_matrix.npy', Q)
    with open(output_dir / 'bit_labels.txt', 'w', encoding='utf-8') as f:
        for i, v in enumerate(variables):
            f.write(f'{i},{v.name}\n')

    summary = {
        'mode': 'hu_qubo',
        'data_dir': str(data_dir),
        'qubo_m': cfg.qubo_m,
        'direct_sh': cfg.direct_sh,
        'direct_su': cfg.direct_su,
        'g': cfg.g,
        's0': cfg.s0,
        'theta': cfg.theta,
        'dx': cfg.dx,
        'dt': cfg.dt,
        'penalty': cfg.penalty,
        'original_binary_variables': len(bit_symbols),
        'auxiliary_variables': len(reducer.aux_vars),
        'total_binary_variables': len(variables),
        'objective_degree_before_reduction': int(sp.Poly(objective, *sorted(objective.free_symbols, key=lambda s: s.name)).total_degree()) if objective.free_symbols else 0,
        'objective_constant': constant,
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    result = {
        'summary': summary,
        'Q': Q,
        'variables': variables,
        'decoded_h': None,
        'decoded_u': None,
        'best_bits': None,
        'best_objective': None,
    }

    if solve:
        best_bits = qubo_bruteforce(Q)
        best_val = float(best_bits @ Q @ best_bits)
        decoded = decode_unknowns(best_bits, variables, unknown_map, cfg)
        decoded_h = fill_decoded_grid(h_grid, decoded, 'h')
        decoded_u = fill_decoded_grid(u_grid, decoded, 'u')
        save_grid_csv(output_dir / 'decoded_h.csv', x_h, t_h, decoded_h)
        save_grid_csv(output_dir / 'decoded_u.csv', x_h, t_h, decoded_u)
        with open(output_dir / 'solution_bits.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(str(int(b)) for b in best_bits) + '\n')
        with open(output_dir / 'solution_summary.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'best_objective': best_val,
                    'decoded_unknowns': {
                        f'{kind}[t={ti},x={xi}]': value for (kind, ti, xi), value in decoded.items()
                    },
                },
                f,
                indent=2,
            )
        result.update(
            {
                'decoded_h': decoded_h,
                'decoded_u': decoded_u,
                'best_bits': best_bits,
                'best_objective': best_val,
            }
        )

    if verbose:
        print('Built direct h-u QUBO')
        for key, value in summary.items():
            print(f'{key}: {value}')
        if solve:
            print(f'best_objective: {result["best_objective"]}')
            print('Decoded unknown values:')
            decoded = json.load(open(output_dir / 'solution_summary.json', 'r', encoding='utf-8'))['decoded_unknowns']
            for key, value in decoded.items():
                print(f'  {key} = {value:.6f}')
        print(f'Saved outputs to: {output_dir}')

    return result


def main():
    parser = argparse.ArgumentParser(description='Direct h-u QUBO research script.')
    parser.add_argument('--data-dir', default='hu_qubo/data')
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--qubo-m', type=int, default=2)
    parser.add_argument('--g', type=float, default=9.81)
    parser.add_argument('--s0', type=float, default=1.0e-3)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--dx', type=float, default=100.0)
    parser.add_argument('--dt', type=float, default=100.0)
    parser.add_argument('--direct-sh', type=float, default=3.5)
    parser.add_argument('--direct-su', type=float, default=3.5)
    parser.add_argument('--penalty', type=float, default=10.0)
    parser.add_argument('--no-solve', action='store_true', help='Only build the QUBO, do not brute-force solve it.')
    args = parser.parse_args()

    cfg = QUBOConfig(
        g=args.g,
        s0=args.s0,
        theta=args.theta,
        dx=args.dx,
        dt=args.dt,
        qubo_m=args.qubo_m,
        direct_sh=args.direct_sh,
        direct_su=args.direct_su,
        penalty=args.penalty,
    )
    output_dir = Path(args.output_root) / 'hu_qubo'
    build_and_optionally_solve(cfg, Path(args.data_dir), output_dir, solve=not args.no_solve, verbose=True)


if __name__ == '__main__':
    main()
