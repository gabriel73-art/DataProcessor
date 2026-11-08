# -*- coding: utf-8 -*-
"""
Comportamiento de los optimizadores propios en funciones de prueba 2D.

  1. Cuadrática mal condicionada y rotada 30°:
         f(w) = 1/2 (w - w*)^T A (w - w*),  número de condición kappa = 20
  2. Función de Rosenbrock:
         f(x, y) = (1 - x)^2 + 100 (y - x^2)^2,  mínimo en (1, 1)

Para cada optimizador se elige la tasa de aprendizaje con una búsqueda en
rejilla logarítmica (la que da menor f tras el presupuesto de iteraciones),
de modo que la comparación sea justa. Se guardan las figuras y una tabla.

Uso:  python trayectorias_2d.py
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimizadores as opt  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
os.makedirs(OUT, exist_ok=True)

OPTIMIZADORES = {
    'GD':             lambda p, lr: opt.SGDMomentum(p, lr=lr),
    'Momentum (0.9)': lambda p, lr: opt.SGDMomentum(p, lr=lr, momentum=0.9),
    'Nesterov (0.9)': lambda p, lr: opt.SGDMomentum(p, lr=lr, momentum=0.9, nesterov=True),
    'Adagrad':        lambda p, lr: opt.Adagrad(p, lr=lr),
    'RMSprop':        lambda p, lr: opt.RMSprop(p, lr=lr),
    'Adam':           lambda p, lr: opt.Adam(p, lr=lr),
}
COLORES = dict(zip(OPTIMIZADORES, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']))
LR_GRID = np.logspace(-5, 1, 25)

# --- Funciones de prueba ----------------------------------------------------
KAPPA, ANG = 20.0, math.radians(30)
R = torch.tensor([[math.cos(ANG), -math.sin(ANG)], [math.sin(ANG), math.cos(ANG)]])
A = R @ torch.diag(torch.tensor([1.0, KAPPA])) @ R.T
W_STAR_Q = torch.tensor([0.0, 0.0])


def cuadratica(w):
    d = w - W_STAR_Q
    return 0.5 * d @ A @ d


def rosenbrock(w):
    return (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2


PROBLEMAS = {
    'cuadratica': dict(f=cuadratica, x0=[-4.0, 3.0], iters=200, w_star=[0.0, 0.0],
                       xlim=(-5, 5), ylim=(-4, 4), titulo=f'Cuadrática rotada (κ = {KAPPA:.0f})'),
    'rosenbrock': dict(f=rosenbrock, x0=[-1.5, 2.0], iters=3000, w_star=[1.0, 1.0],
                       xlim=(-2, 2), ylim=(-1, 3), titulo='Rosenbrock'),
}


def run(f, x0, make_opt, lr, iters):
    w = torch.tensor(x0, requires_grad=True)
    o = make_opt([w], lr)
    path, vals = [w.detach().clone()], []
    for _ in range(iters):
        o.zero_grad()
        loss = f(w)
        vals.append(loss.item())
        loss.backward()
        o.step()
        if not torch.isfinite(w).all() or w.abs().max() > 1e6:
            return None, None
        path.append(w.detach().clone())
    vals.append(f(w).item())
    return torch.stack(path).numpy(), np.array(vals)


def mejor_lr(f, x0, make_opt, iters):
    mejor = (math.inf, None)
    for lr in LR_GRID:
        _, vals = run(f, x0, make_opt, lr, iters)
        if vals is not None and vals[-1] < mejor[0]:
            mejor = (vals[-1], lr)
    return mejor[1]


def main():
    filas = []
    for nombre_p, P in PROBLEMAS.items():
        f, x0, iters = P['f'], P['x0'], P['iters']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

        xs = np.linspace(*P['xlim'], 300)
        ys = np.linspace(*P['ylim'], 300)
        Z = np.array([[f(torch.tensor([x, y])).item() for x in xs] for y in ys])
        ax1.contour(xs, ys, np.log10(Z + 1e-12), levels=30, cmap='Greys', linewidths=0.6)
        # zorder alto: el óptimo y el inicio se dibujan encima de las trayectorias
        ax1.plot(*P['w_star'], '*', ms=18, mfc='gold', mec='black', mew=1.2,
                 zorder=10, label='Óptimo')
        ax1.plot(*x0, 'o', ms=8, mfc='black', mec='white', zorder=10, label='Inicio')

        for nombre_o, make in OPTIMIZADORES.items():
            lr = mejor_lr(f, x0, make, iters)
            path, vals = run(f, x0, make, lr, iters)
            dist = np.linalg.norm(path - np.array(P['w_star']), axis=1)
            hit = np.where(dist < 1e-3)[0]
            filas.append(dict(problema=nombre_p, optimizador=nombre_o, lr=f"{lr:.2e}",
                              f_final=f"{vals[-1]:.3e}", dist_final=f"{dist[-1]:.3e}",
                              iter_dist_1e3=int(hit[0]) if len(hit) else 'no alcanza'))
            c = COLORES[nombre_o]
            ax1.plot(path[:, 0], path[:, 1], '-', color=c, lw=1.3, label=f'{nombre_o} (η={lr:.1e})')
            ax2.semilogy(np.maximum(vals, 1e-16), color=c, lw=1.5, label=nombre_o)
            print(f"{nombre_p:11s} {nombre_o:15s} lr={lr:.2e}  f_final={vals[-1]:.3e}  "
                  f"dist={dist[-1]:.2e}  iter(dist<1e-3)={filas[-1]['iter_dist_1e3']}")

        ax1.set_xlim(*P['xlim'])
        ax1.set_ylim(*P['ylim'])
        ax1.set_title(f"{P['titulo']}: trayectorias")
        ax1.legend(fontsize=8, loc='lower right' if nombre_p == 'rosenbrock' else 'upper right')
        ax2.set_title(f"{P['titulo']}: f(w) por iteración")
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('f(w)  (escala log)')
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'trayectorias_{nombre_p}.png'), dpi=130)
        plt.close(fig)

    with open(os.path.join(OUT, 'trayectorias_2d.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nFiguras y tabla guardadas en {OUT}")


if __name__ == '__main__':
    main()
