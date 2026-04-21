from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



def load_case_data(data_dir: Path):
    geom = np.loadtxt(data_dir / 'geometry-clipped.csv', delimiter=',', skiprows=1)
    hydro = np.loadtxt(data_dir / 'hydrograph.csv', delimiter=',', skiprows=1)
    x, z = geom[:, 0], geom[:, 1]
    hydro_t, hydro_q = hydro[:, 0], hydro[:, 1]
    return x, z, hydro_t, hydro_q



def build_qin(hydro_t, hydro_q):
    def qin(t):
        return np.interp(t, hydro_t, hydro_q)

    return qin



def normalize_mode(mode: str) -> str:
    aliases = {
        'qubo': 'newton_qubo',
        'qubo_with_newton': 'newton_qubo',
        'linear_qubo': 'newton_qubo',
    }
    return aliases.get(mode, mode)



def get_default_max_iter(mode: str) -> int:
    mode = normalize_mode(mode)
    if mode == 'hhl':
        return 5
    if mode == 'direct_qubo':
        return 1
    return 50



def solve_linear_system(mode: str, J: np.ndarray, rhs: np.ndarray, qubo_m: int, qubo_s: float):
    mode = normalize_mode(mode)
    if mode == 'classical':
        return np.linalg.solve(J, rhs)
    if mode == 'hhl':
        try:
            from hhl_solver import solve_qls
        except Exception as exc:
            raise RuntimeError(
                'HHL mode requires the older Qiskit environment and local quantum_linear_solvers install.'
            ) from exc
        return solve_qls(J, rhs)
    if mode == 'newton_qubo':
        from newton_qubo_solver import linear_to_qubo_solve

        return linear_to_qubo_solve(J, rhs, m=qubo_m, s=qubo_s)
    raise ValueError(f'Unknown mode: {mode}')



def normal_depth_rectangular(
    Q_target: float,
    b: float,
    n_mann: float,
    S0: float,
    y_low: float = 1e-6,
    y_high: float = 100.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    S0_eff = abs(float(S0))
    if S0_eff <= 0.0:
        raise ValueError('Normal depth requires a positive bed slope magnitude.')

    Q_target = abs(float(Q_target))

    def discharge_from_depth(y: float) -> float:
        area = b * y
        wetted_perimeter = b + 2.0 * y
        hydraulic_radius = area / wetted_perimeter
        return (1.0 / n_mann) * area * (hydraulic_radius ** (2.0 / 3.0)) * (S0_eff ** 0.5)

    while discharge_from_depth(y_high) < Q_target:
        y_high *= 2.0
        if y_high > 1e6:
            raise RuntimeError('Failed to bracket normal depth.')

    for _ in range(max_iter):
        y_mid = 0.5 * (y_low + y_high)
        q_mid = discharge_from_depth(y_mid)
        if abs(q_mid - Q_target) < tol:
            return y_mid
        if q_mid < Q_target:
            y_low = y_mid
        else:
            y_high = y_mid

    return 0.5 * (y_low + y_high)



def solve_swe(
    mode: str = 'classical',
    data_dir: Path | str = 'data',
    nonlinear_tol: float = 1e-2,
    max_newton_iter: int | None = None,
    qubo_m: int = 2,
    qubo_s: float = 1.0,
    direct_sq: float = 40.0,
    direct_sy: float = 3.5,
    verbose: bool = True,
):
    mode = normalize_mode(mode)
    data_dir = Path(data_dir)
    x, z, hydro_t, hydro_q = load_case_data(data_dir)
    qin = build_qin(hydro_t, hydro_q)

    g = 32.2
    b = 10.0
    n_mann = 0.025
    theta = 0.8
    dx = x[1] - x[0]
    dt = hydro_t[1] - hydro_t[0]
    N = len(x)
    S0 = (z[0] - z[-1]) / (x[-1] - x[0])
    max_newton_iter = get_default_max_iter(mode) if max_newton_iter is None else max_newton_iter

    Q = np.full(N, hydro_q[0], dtype=float)
    y0_normal = normal_depth_rectangular(hydro_q[0], b, n_mann, S0)
    y = np.full(N, y0_normal, dtype=float)

    direct_cfg = None
    if mode == 'direct_qubo':
        from direct_qubo_system import DirectQUBOConfig

        direct_cfg = DirectQUBOConfig(
            g=g,
            b=b,
            n_mann=n_mann,
            theta=theta,
            mQ=qubo_m,
            my=qubo_m,
            sQ=direct_sq,
            sy=direct_sy,
        )

    def A(depth):
        return b * depth

    def P(depth):
        return b + 2.0 * depth

    def R(depth):
        return A(depth) / P(depth)

    def Sf(discharge, depth):
        return n_mann**2 * discharge * np.abs(discharge) / (A(depth) ** 2 * R(depth) ** (4.0 / 3.0))

    def dSf_dQ(discharge, depth):
        return 2.0 * n_mann**2 * np.abs(discharge) / (A(depth) ** 2 * R(depth) ** (4.0 / 3.0))

    def dSf_dy(discharge, depth):
        sf_val = Sf(discharge, depth)
        return sf_val * (-2.0 / depth - (4.0 / 3.0) * b / (depth * (b + 2.0 * depth)))

    def storage(depth):
        return np.sum(A(depth)) * dx

    def mass_step_error(Sn1, Sn, Qin_n, Qin_n1, Qout_n, Qout_n1, dt_local):
        Qin_avg = 0.5 * (Qin_n + Qin_n1)
        Qout_avg = 0.5 * (Qout_n + Qout_n1)
        return (Sn1 - Sn) - dt_local * (Qin_avg - Qout_avg)

    def residual(Qn, yn, Qc, yc, t):
        Rv = [Qc[0] - qin(t)]
        for i in range(N - 1):
            j = i + 1
            Ai, Aj = A(yc[i]), A(yc[j])
            Ain, Ajn = A(yn[i]), A(yn[j])
            C = (Ai + Aj - Ain - Ajn) / (2.0 * dt)
            C += theta * (Qc[j] - Qc[i]) / dx
            C += (1.0 - theta) * (Qn[j] - Qn[i]) / dx
            Rv.append(C)

            M = (Qc[j] + Qc[i] - Qn[j] - Qn[i]) / (2.0 * dt)
            K_new = (Qc[j] ** 2 / Aj - Qc[i] ** 2 / Ai) / dx
            P_new = g * (Ai + Aj) / 2.0 * (yc[j] - yc[i]) / dx
            F_new = g * (Ai + Aj) / 2.0 * (Sf(Qc[j], yc[j]) + Sf(Qc[i], yc[i])) / 2.0
            S_new = -g * (Ai + Aj) / 2.0 * S0

            K_old = (Qn[j] ** 2 / Ajn - Qn[i] ** 2 / Ain) / dx
            P_old = g * (Ain + Ajn) / 2.0 * (yn[j] - yn[i]) / dx
            F_old = g * (Ain + Ajn) / 2.0 * (Sf(Qn[j], yn[j]) + Sf(Qn[i], yn[i])) / 2.0
            S_old = -g * (Ain + Ajn) / 2.0 * S0

            M += theta * (K_new + P_new + F_new + S_new)
            M += (1.0 - theta) * (K_old + P_old + F_old + S_old)
            Rv.append(M)

        Rv.append(Sf(Qc[-1], yc[-1]) - S0)
        return np.array(Rv, dtype=float)

    def jacobian(Qn, yn, Qc, yc, t):
        J = np.zeros((2 * N, 2 * N), dtype=float)
        J[0, 0] = 1.0
        row = 1
        for i in range(N - 1):
            j = i + 1
            Ai, Aj = A(yc[i]), A(yc[j])
            Abar = 0.5 * (Ai + Aj)
            Sfi, Sfj = Sf(Qc[i], yc[i]), Sf(Qc[j], yc[j])

            J[row, i] = -theta / dx
            J[row, j] = theta / dx
            J[row, N + i] = b / (2.0 * dt)
            J[row, N + j] = b / (2.0 * dt)
            row += 1

            J[row, i] = 1.0 / (2.0 * dt) + theta * (
                -(2.0 * Qc[i] / Ai) / dx + g * (Abar / 4.0) * dSf_dQ(Qc[i], yc[i])
            )
            J[row, j] = 1.0 / (2.0 * dt) + theta * (
                (2.0 * Qc[j] / Aj) / dx + g * (Abar / 4.0) * dSf_dQ(Qc[j], yc[j])
            )

            dconv_dyi = (b * Qc[i] ** 2) / (dx * Ai ** 2)
            dconv_dyj = -(b * Qc[j] ** 2) / (dx * Aj ** 2)
            dpress_dyi = g * ((b / 2.0) * (yc[j] - yc[i]) / dx - (Ai + Aj) / (2.0 * dx))
            dpress_dyj = g * ((b / 2.0) * (yc[j] - yc[i]) / dx + (Ai + Aj) / (2.0 * dx))
            dfric_dyi = g * ((b / 4.0) * (Sfi + Sfj) + (Abar / 4.0) * dSf_dy(Qc[i], yc[i]))
            dfric_dyj = g * ((b / 4.0) * (Sfi + Sfj) + (Abar / 4.0) * dSf_dy(Qc[j], yc[j]))
            dslope_dy = -g * (b / 2.0) * S0

            J[row, N + i] = theta * (dconv_dyi + dpress_dyi + dfric_dyi + dslope_dy)
            J[row, N + j] = theta * (dconv_dyj + dpress_dyj + dfric_dyj + dslope_dy)
            row += 1

        J[row, N - 1] = dSf_dQ(Qc[-1], yc[-1])
        J[row, 2 * N - 1] = dSf_dy(Qc[-1], yc[-1])
        return J

    def newton_step(Qn, yn, Qc, yc, t):
        r = residual(Qn, yn, Qc, yc, t)
        J = jacobian(Qn, yn, Qc, yc, t)
        if verbose:
            print('Newton step linear system: J dU = -r')
            print('Condition number of J:', np.linalg.cond(J))
            print('Norm of residual ||r||_inf:', np.linalg.norm(r, ord=np.inf))
        dU = solve_linear_system(mode, J, -r, qubo_m, qubo_s)
        vec = np.r_[Qc, yc] + dU
        return vec[:N], vec[N:], np.linalg.norm(dU, ord=np.inf)

    Tmax = hydro_t[-1]
    nt = int(Tmax / dt)
    Q_hist = [Q.copy()]
    y_hist = [y.copy()]
    t_hist = [0.0]
    S_hist = [storage(y)]
    mass_err_step = []
    mass_err_cum = [0.0]

    for nstep in range(nt):
        t_old = nstep * dt
        t_new = (nstep + 1) * dt
        Q_old = Q.copy()
        y_old = y.copy()
        S_old = storage(y_old)
        Qin_old = qin(t_old)
        Qout_old = Q_old[-1]
        Qg, yg = Q.copy(), y.copy()

        if verbose:
            print('=' * 49)
            print(f'Time step {nstep + 1}: t = {t_new / 60.0:.2f} min')
            print('=' * 49)

        if mode == 'direct_qubo':
            from direct_qubo_solver import solve_direct_qubo_step

            direct_result = solve_direct_qubo_step(
                x,
                z,
                Q_old,
                y_old,
                float(qin(t_new)),
                float(dt),
                cfg=direct_cfg,
                use_friction=True,
                verbose=verbose,
            )
            Q, y = direct_result.Q, direct_result.y
        else:
            for k in range(max_newton_iter):
                r = residual(Q, y, Qg, yg, t_new)
                error = np.linalg.norm(r, ord=np.inf)
                if verbose:
                    print(f'Newton iteration {k + 1}: residual = {error:.4e}')
                if error <= nonlinear_tol:
                    break
                Qg, yg, _ = newton_step(Q, y, Qg, yg, t_new)
            Q, y = Qg, yg

        S_new = storage(y)
        Qin_new = qin(t_new)
        Qout_new = Q[-1]
        e = mass_step_error(S_new, S_old, Qin_old, Qin_new, Qout_old, Qout_new, dt)

        mass_err_step.append(e)
        mass_err_cum.append(mass_err_cum[-1] + e)
        S_hist.append(S_new)
        Q_hist.append(Q.copy())
        y_hist.append(y.copy())
        t_hist.append(t_new)

    return {
        'mode': mode,
        'x': x,
        'z': z,
        't_min': np.array(t_hist) / 60.0,
        'Q_hist': np.array(Q_hist),
        'y_hist': np.array(y_hist),
        'S_hist': np.array(S_hist),
        'mass_err_step': np.array(mass_err_step),
        'mass_err_cum': np.array(mass_err_cum),
        'qin': qin,
        'hydro_t_min': hydro_t / 60.0,
        'hydro_q': hydro_q,
        'peak_inflow_time_min': float(hydro_t[np.argmax(hydro_q)] / 60.0),
        'initial_depth_normal': float(y0_normal),
    }



def save_time_space_csv(path: Path, t_min: np.ndarray, x: np.ndarray, data: np.ndarray):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('t\\x,' + ','.join(f'{xi:.0f}' for xi in x) + '\n')
        for ti, row in zip(t_min, data):
            f.write(f'{ti:.0f},' + ','.join(f'{v:.5f}' for v in row) + '\n')


def save_mass_balance_csv(path: Path, result: dict):
    qin = result['qin']
    t_min = result['t_min']
    Q_hist = result['Q_hist']
    S_hist = result['S_hist']
    mass_err_step = result['mass_err_step']
    mass_err_cum = result['mass_err_cum']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('t_sec,storage,Qin,Qout,step_error,cum_error\n')
        for k in range(len(t_min)):
            tsec = t_min[k] * 60.0
            Qin_k = qin(tsec)
            Qout_k = Q_hist[k, -1]
            step_e = 0.0 if k == 0 else mass_err_step[k - 1]
            f.write(
                f'{tsec:.6f},{S_hist[k]:.10e},{Qin_k:.10e},{Qout_k:.10e},{step_e:.10e},{mass_err_cum[k]:.10e}\n'
            )


def save_outputs(result: dict, output_dir: Path, prefix: str = ''):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{prefix}_' if prefix else ''
    save_time_space_csv(output_dir / f'{stem}output_Q.csv', result['t_min'], result['x'], result['Q_hist'])
    save_time_space_csv(output_dir / f'{stem}output_depth.csv', result['t_min'], result['x'], result['y_hist'])
    save_mass_balance_csv(output_dir / f'{stem}mass_balance.csv', result)



def configure_plot_style():
    mpl.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.4,
            'lines.markersize': 4,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )



def create_time_plot():
    return plt.subplots(figsize=(5.2, 4.0))



def create_space_plot():
    return plt.subplots(figsize=(6.6, 3.2))



def get_gradient_colors(count: int, cmap_name: str, vmin: float = 0.20, vmax: float = 0.90):
    cmap = mpl.cm.get_cmap(cmap_name)
    if count <= 1:
        return [cmap(0.6)]
    values = np.linspace(vmin, vmax, count)
    return [cmap(v) for v in values]



def choose_label_index(x: np.ndarray, side: str = 'right') -> int:
    if len(x) < 3:
        return len(x) - 1
    return max(0, len(x) - 2) if side == 'right' else 1



def add_inline_line_label(ax, x, y, text, color, side='right', dy_frac=0.012, fontsize=9):
    idx = choose_label_index(x, side=side)
    x_span = max(float(np.max(x) - np.min(x)), 1.0)
    y_span = max(float(np.max(y) - np.min(y)), 1.0)
    x_offset = 0.015 * x_span * (-1.0 if side == 'right' else 1.0)
    y_offset = dy_frac * y_span
    ax.text(
        float(x[idx]) + x_offset,
        float(y[idx]) + y_offset,
        text,
        color=color,
        fontsize=fontsize,
        ha='right' if side == 'right' else 'left',
        va='bottom',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.2),
    )



def finalize_plot(ax, xlabel, ylabel, legend=True, ncol=1):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(frameon=False, ncol=ncol)
    plt.tight_layout()



def get_profile_times(t_hist: np.ndarray, count: int = 6):
    return np.linspace(0.0, float(t_hist[-1]), count)



def get_split_profile_times(t_hist: np.ndarray, peak_time: float, count_each: int = 4):
    eps = 1e-9
    before = np.linspace(0.0, peak_time, count_each)
    if peak_time >= float(t_hist[-1]) - eps:
        after = np.array([peak_time])
    else:
        after = np.linspace(peak_time, float(t_hist[-1]), count_each)
    before = np.unique(np.round(before, 8))
    after = np.unique(np.round(after, 8))
    return before, after



def plot_hydraulic_grade_line_panels(fig, axes, x, z, t_hist, y_hist, peak_time, reference=None):
    time_sets = get_split_profile_times(t_hist, peak_time, count_each=4)
    panel_titles = [
        f'Hydraulic Grade Line: 0 to inflow peak ({peak_time:.1f} min)',
        f'Hydraulic Grade Line: after inflow peak ({peak_time:.1f} min to {float(t_hist[-1]):.1f} min)',
    ]
    cmap_names = ['Blues', 'Oranges']

    for ax, times_local, title, cmap_name in zip(axes, time_sets, panel_titles, cmap_names):
        colors = get_gradient_colors(len(times_local), cmap_name)
        ax.plot(x, z, color='black', linewidth=2.2)
        add_inline_line_label(ax, x, z, 'Channel bed', 'black', side='left', dy_frac=-0.02)

        for k, tp in enumerate(times_local):
            i = int(np.argmin(np.abs(t_hist - tp)))
            color = colors[k]
            hgl = z + y_hist[i]
            ax.plot(x, hgl, color=color)
            side = 'left' if k % 2 == 0 else 'right'
            add_inline_line_label(ax, x, hgl, f'{int(round(t_hist[i]))} min', color, side=side)
            if reference is not None:
                ax.plot(x, z + reference['y_hist'][i], '--', linewidth=1.1, color=color, alpha=0.75)

        ax.set_xlim(x[0], x[-1])
        ax.set_title(title)
        ax.set_xlabel('Distance (ft)')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Hydraulic Grade Line (ft)')
    if reference is not None:
        axes[1].plot([], [], '--', color='0.35', linewidth=1.1, label='Classical reference')
        axes[1].legend(frameon=False, loc='lower right')
    fig.tight_layout()



def plot_results(result: dict, output_dir: Path, reference: dict | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    x = result['x']
    z = result['z']
    t_hist = result['t_min']
    Q_hist = result['Q_hist']
    y_hist = result['y_hist']
    mass_err_cum = result['mass_err_cum']
    peak_time = result['peak_inflow_time_min']
    N = len(x)
    mid = N // 2
    times_plot = get_profile_times(t_hist, count=6)
    mode_label = result['mode'].upper() if result['mode'] != 'classical' else 'Classical'

    fig, ax = create_time_plot()
    locations = [(0, 'Upstream'), (mid, 'Mid-channel'), (-1, 'Downstream')]
    colors = get_gradient_colors(len(locations), 'viridis')
    for (idx, label), color in zip(locations, colors):
        ax.plot(t_hist, Q_hist[:, idx], marker='+', color=color, label=f'{label} ({mode_label})')
        if reference is not None:
            ax.plot(t_hist, reference['Q_hist'][:, idx], '--', linewidth=1.2, color=color, alpha=0.75, label=f'{label} (Classical)')
    finalize_plot(ax, 'Time (min)', 'Discharge (cfs)', legend=True)
    plt.savefig(output_dir / 'hydrographs.pdf')
    plt.close(fig)

    fig, ax = create_space_plot()
    colors = get_gradient_colors(len(times_plot), 'viridis')
    dashed_labeled = False
    for tp, color in zip(times_plot, colors):
        i = int(np.argmin(np.abs(t_hist - tp)))
        ax.plot(x, Q_hist[i], color=color, label=f't = {int(round(t_hist[i]))} min')
        if reference is not None:
            ax.plot(
                x,
                reference['Q_hist'][i],
                '--',
                linewidth=1.1,
                color=color,
                alpha=0.75,
                label='Classical reference' if not dashed_labeled else None,
            )
            dashed_labeled = True
    ax.set_xlim(x[0], x[-1])
    finalize_plot(ax, 'Distance (ft)', 'Discharge (cfs)', legend=True, ncol=2)
    plt.savefig(output_dir / 'discharge_profiles.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), sharey=True)
    plot_hydraulic_grade_line_panels(fig, axes, x, z, t_hist, y_hist, peak_time, reference=reference)
    plt.savefig(output_dir / 'water_surface_profiles.pdf')
    plt.close(fig)

    fig, ax = create_space_plot()
    peak_color = get_gradient_colors(1, 'magma')[0]
    ax.plot(x, np.max(Q_hist, axis=0), 'o-', color=peak_color, label=f'Peak discharge ({mode_label})')
    if reference is not None:
        ax.plot(x, np.max(reference['Q_hist'], axis=0), '--', linewidth=1.2, color=peak_color, alpha=0.75, label='Peak discharge (Classical)')
    ax.set_xlim(x[0], x[-1])
    finalize_plot(ax, 'Distance (ft)', 'Peak Discharge (cfs)', legend=True)
    plt.savefig(output_dir / 'peak_attenuation.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    mass_color = get_gradient_colors(1, 'cividis')[0]
    ax.plot(t_hist, mass_err_cum, color=mass_color, label=f'Cumulative error ({mode_label})')
    if reference is not None:
        ax.plot(reference['t_min'], reference['mass_err_cum'], '--', linewidth=1.2, color=mass_color, alpha=0.75, label='Cumulative error (Classical)')
    finalize_plot(ax, 'Time (min)', 'Cumulative Volume Error (ft³)', legend=True)
    plt.savefig(output_dir / 'mass_balance.pdf')
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description='1D shallow water routing with classical, HHL, Newton-QUBO, or direct nonlinear QUBO solves.'
    )
    parser.add_argument(
        '--mode',
        choices=['classical', 'hhl', 'direct_qubo', 'newton_qubo', 'qubo_with_newton', 'qubo', 'linear_qubo'],
        default='classical',
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--nonlinear-tol', type=float, default=1e-2)
    parser.add_argument('--max-newton-iter', type=int, default=None)
    parser.add_argument('--qubo-m', type=int, default=3)
    parser.add_argument('--qubo-s', type=float, default=1.0)
    parser.add_argument('--direct-sq', type=float, default=40.0)
    parser.add_argument('--direct-sy', type=float, default=3.5)
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    output_dir = Path(args.output_root) / mode
    result = solve_swe(
        mode=mode,
        data_dir=args.data_dir,
        nonlinear_tol=args.nonlinear_tol,
        max_newton_iter=args.max_newton_iter,
        qubo_m=args.qubo_m,
        qubo_s=args.qubo_s,
        direct_sq=args.direct_sq,
        direct_sy=args.direct_sy,
        verbose=True,
    )
    save_outputs(result, output_dir)

    reference = None
    if mode != 'classical':
        reference = solve_swe(
            mode='classical',
            data_dir=args.data_dir,
            nonlinear_tol=args.nonlinear_tol,
            max_newton_iter=args.max_newton_iter,
            qubo_m=args.qubo_m,
            qubo_s=args.qubo_s,
            direct_sq=args.direct_sq,
            direct_sy=args.direct_sy,
            verbose=False,
        )
        save_outputs(reference, output_dir, prefix='classical_reference')

    plot_results(result, output_dir, reference=reference)
    print(f'Saved results to: {output_dir}')



if __name__ == '__main__':
    main()
