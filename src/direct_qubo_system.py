from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp


@dataclass
class DirectQUBOConfig:
    g: float = 32.2
    b: float = 10.0
    n_mann: float = 0.025
    theta: float = 0.8
    mQ: int = 2
    my: int = 2
    sQ: float = 40.0
    sy: float = 3.5


def _sx(value):
    """Convert numeric input to a stable SymPy scalar.

    Using exact symbolic scalars lets SymPy cancel area factors cleanly after
    the momentum equation is rescaled to remove 1/A and 1/A^2 terms.
    """
    return sp.nsimplify(value)



def area_rect(depth, cfg: DirectQUBOConfig):
    return _sx(cfg.b) * depth



def hydraulic_radius_rect(depth, cfg: DirectQUBOConfig):
    b = _sx(cfg.b)
    d = _sx(depth)
    return (b * d) / (b + 2 * d)



def momentum_polynomial_scale(Ai, Aj):
    """Scale factor that clears all new-time denominators.

    The convective term contributes Q^2/A and Manning friction contributes
    Q^2/A^2. Multiplying the full momentum residual by A_i^2 A_j^2 eliminates
    those symbolic denominators before the residual is squared.
    """
    return Ai**2 * Aj**2



def build_direct_nonlinear_objective(
    x: np.ndarray,
    z: np.ndarray,
    Qn: np.ndarray,
    yn: np.ndarray,
    Qin_np1: float,
    dt: float,
    cfg: DirectQUBOConfig,
    encode,
    *,
    use_friction: bool = True,
):
    """Build a one-step polynomial direct nonlinear binary objective.

    The inflow discharge at the new time level is fixed to the hydrograph
    value and the upstream depth is carried from the previous time level. The
    remaining discharge and depth values are represented with binary encodings.

    To avoid a rational objective in the decision variables, the momentum
    residual is multiplied by A_i^2 A_j^2 before squaring. The old-time terms
    remain numeric constants and do not affect polynomiality in the binaries.
    """

    N = len(x)
    dx = _sx(float(x[1] - x[0]))
    dt_s = _sx(float(dt))
    S0 = _sx(float((z[0] - z[-1]) / (x[-1] - x[0])))
    g = _sx(cfg.g)
    theta = _sx(cfg.theta)
    n_sq = _sx(cfg.n_mann) ** 2
    half = sp.Rational(1, 2)

    Q = [None] * N
    y = [None] * N

    Q[0] = _sx(Qin_np1)
    y[0] = _sx(yn[0])

    for i in range(1, N):
        Q[i] = encode('Q', i, cfg.mQ, cfg.sQ)
        y[i] = encode('y', i, cfg.my, cfg.sy)

    obj = sp.Integer(0)

    for i in range(N - 1):
        j = i + 1

        Ai = area_rect(y[i], cfg)
        Aj = area_rect(y[j], cfg)
        Ain = _sx(cfg.b) * _sx(yn[i])
        Ajn = _sx(cfg.b) * _sx(yn[j])

        continuity = (Ai + Aj - Ain - Ajn) / (2 * dt_s)
        continuity += theta * (Q[j] - Q[i]) / dx
        continuity += (1 - theta) * (_sx(Qn[j]) - _sx(Qn[i])) / dx

        Knew = (Q[j] ** 2 / Aj - Q[i] ** 2 / Ai) / dx
        Kold = (_sx(Qn[j]) ** 2 / Ajn - _sx(Qn[i]) ** 2 / Ain) / dx

        Pnew = g * (Ai + Aj) * (y[j] - y[i]) / (2 * dx)
        Pold = g * (Ain + Ajn) * (_sx(yn[j]) - _sx(yn[i])) / (2 * dx)

        if use_friction:
            Ri_old = hydraulic_radius_rect(yn[i], cfg)
            Rj_old = hydraulic_radius_rect(yn[j], cfg)
            Ri_scale = Ri_old ** _sx(sp.Rational(4, 3))
            Rj_scale = Rj_old ** _sx(sp.Rational(4, 3))

            Sfi = n_sq * Q[i] ** 2 / (Ai ** 2 * Ri_scale)
            Sfj = n_sq * Q[j] ** 2 / (Aj ** 2 * Rj_scale)
            Sfi_old = n_sq * _sx(Qn[i]) ** 2 / (Ain ** 2 * Ri_scale)
            Sfj_old = n_sq * _sx(Qn[j]) ** 2 / (Ajn ** 2 * Rj_scale)
            Fnew = g * (Ai + Aj) * (Sfi + Sfj) / 4
            Fold = g * (Ain + Ajn) * (Sfi_old + Sfj_old) / 4
        else:
            Fnew = sp.Integer(0)
            Fold = sp.Integer(0)

        Snew = -g * (Ai + Aj) * S0 / 2
        Sold = -g * (Ain + Ajn) * S0 / 2

        momentum = (Q[j] + Q[i] - _sx(Qn[j]) - _sx(Qn[i])) / (2 * dt_s)
        momentum += theta * (Knew + Pnew + Fnew + Snew)
        momentum += (1 - theta) * (Kold + Pold + Fold + Sold)

        # Clear new-time denominators before squaring so the objective stays
        # polynomial in the binary variables.
        momentum_poly = sp.expand(sp.cancel(momentum_polynomial_scale(Ai, Aj) * momentum))
        obj += sp.expand(continuity**2 + momentum_poly**2)

    return sp.expand(obj), Q, y
