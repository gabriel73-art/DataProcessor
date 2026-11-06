# -*- coding: utf-8 -*-
"""
Validación numérica de los optimizadores propios contra torch.optim.

Para cada algoritmo se entrena la misma red pequeña (CNN), con la misma
inicialización y los mismos datos aleatorios, una vez con nuestra
implementación y otra con la de PyTorch. Si las ecuaciones son correctas,
los parámetros tras N pasos deben coincidir hasta el error de redondeo.

Se usa float64 para que la comparación sea estricta.
Uso:  python test_optimizadores.py
"""
import copy
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimizadores as opt  # noqa: E402

torch.set_default_dtype(torch.float64)
N_STEPS = 200
TOL = 1e-10


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(8, 4),
    )


# (nombre, constructor propio, constructor de PyTorch)
CASOS = [
    ("SGD",
     lambda p: opt.SGDMomentum(p, lr=0.05),
     lambda p: torch.optim.SGD(p, lr=0.05)),
    ("SGD + momentum",
     lambda p: opt.SGDMomentum(p, lr=0.05, momentum=0.9),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    ("Nesterov",
     lambda p: opt.SGDMomentum(p, lr=0.05, momentum=0.9, nesterov=True),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
    ("SGD + momentum + L2",
     lambda p: opt.SGDMomentum(p, lr=0.05, momentum=0.9, weight_decay=1e-3),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-3)),
    ("Adagrad",
     lambda p: opt.Adagrad(p, lr=0.05),
     lambda p: torch.optim.Adagrad(p, lr=0.05)),
    ("RMSprop",
     lambda p: opt.RMSprop(p, lr=1e-3),
     lambda p: torch.optim.RMSprop(p, lr=1e-3)),
    ("Adam",
     lambda p: opt.Adam(p, lr=1e-3),
     lambda p: torch.optim.Adam(p, lr=1e-3)),
    ("Adam + L2",
     lambda p: opt.Adam(p, lr=1e-3, weight_decay=1e-2),
     lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=1e-2)),
    ("AdamW",
     lambda p: opt.AdamW(p, lr=1e-3, weight_decay=1e-2),
     lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
]


def train(model, optimizer, X, y):
    losses = []
    crit = nn.CrossEntropyLoss()
    for i in range(N_STEPS):
        idx = slice((i * 16) % len(X), (i * 16) % len(X) + 16)   # mini-batches de 16
        optimizer.zero_grad()
        loss = crit(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main():
    torch.manual_seed(0)
    X = torch.randn(256, 1, 16, 16)
    y = torch.randint(0, 4, (256,))
    base = make_model()

    filas, ok_total = [], True
    print(f"{'Optimizador':22s} {'max |Δθ|':>12s} {'|Δloss| final':>14s}  resultado")
    for nombre, propio, ref in CASOS:
        m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
        l1 = train(m1, propio(m1.parameters()), X, y)
        l2 = train(m2, ref(m2.parameters()), X, y)
        diff = max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
        dloss = abs(l1[-1] - l2[-1])
        ok = diff < TOL
        ok_total &= ok
        print(f"{nombre:22s} {diff:12.2e} {dloss:14.2e}  {'OK' if ok else 'FALLA'}")
        filas.append(dict(optimizador=nombre, max_diff_parametros=f"{diff:.3e}",
                          diff_loss_final=f"{dloss:.3e}", loss_inicial=f"{l1[0]:.4f}",
                          loss_final=f"{l1[-1]:.4f}", resultado='OK' if ok else 'FALLA'))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados', 'validacion_optimizadores.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{N_STEPS} pasos por optimizador, tolerancia {TOL:g}. Resultados en {out}")
    sys.exit(0 if ok_total else 1)


if __name__ == '__main__':
    main()
