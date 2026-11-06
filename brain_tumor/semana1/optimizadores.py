# -*- coding: utf-8 -*-
"""
Implementación propia de algoritmos de optimización de primer orden.

Cada clase hereda de torch.optim.Optimizer y sigue las ecuaciones del
algoritmo original, con los mismos hiperparámetros por defecto que PyTorch
para poder validarlas contra torch.optim (ver test_optimizadores.py).

Notación: g_t = gradiente, theta = parámetros, eta = tasa de aprendizaje.
"""
import torch
from torch.optim import Optimizer


class SGDMomentum(Optimizer):
    """Descenso de gradiente estocástico con momento (heavy ball) o Nesterov.

        v_t     = mu * v_{t-1} + g_t                 (v_1 = g_1)
        clásico: theta_t = theta_{t-1} - eta * v_t
        Nesterov: theta_t = theta_{t-1} - eta * (g_t + mu * v_t)

    Con momentum=0 se reduce a SGD simple: theta_t = theta_{t-1} - eta * g_t.
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, weight_decay=0.0):
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov requiere momentum > 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p                      # regularización L2
                if mu != 0:
                    state = self.state[p]
                    if 'v' not in state:
                        state['v'] = g.clone()
                    else:
                        state['v'].mul_(mu).add_(g)
                    v = state['v']
                    g = g + mu * v if group['nesterov'] else v
                p.add_(g, alpha=-lr)
        return loss


class Adagrad(Optimizer):
    """Adagrad (Duchi et al., 2011): tasa de aprendizaje adaptativa por parámetro.

        G_t     = G_{t-1} + g_t^2
        theta_t = theta_{t-1} - eta * g_t / (sqrt(G_t) + eps)
    """

    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g + group['weight_decay'] * p
                state = self.state[p]
                if 'G' not in state:
                    state['G'] = torch.zeros_like(p)
                state['G'].addcmul_(g, g)
                p.addcdiv_(g, state['G'].sqrt().add_(group['eps']), value=-group['lr'])
        return loss


class RMSprop(Optimizer):
    """RMSprop (Hinton, 2012): media móvil exponencial de g^2.

        s_t     = alpha * s_{t-1} + (1 - alpha) * g_t^2
        theta_t = theta_{t-1} - eta * g_t / (sqrt(s_t) + eps)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g + group['weight_decay'] * p
                state = self.state[p]
                if 's' not in state:
                    state['s'] = torch.zeros_like(p)
                state['s'].mul_(alpha).addcmul_(g, g, value=1 - alpha)
                p.addcdiv_(g, state['s'].sqrt().add_(group['eps']), value=-group['lr'])
        return loss


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) y AdamW (Loshchilov & Hutter, 2019).

        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat   = m_t / (1 - beta1^t)          (corrección de sesgo)
        v_hat   = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - eta * m_hat / (sqrt(v_hat) + eps)

    decoupled=False: weight decay como término L2 en el gradiente (Adam).
    decoupled=True:  weight decay aplicado directamente a los pesos (AdamW):
                     theta_{t-1} <- theta_{t-1} * (1 - eta * lambda).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    if group['decoupled']:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g + wd * p
                state = self.state[p]
                if 't' not in state:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                state['t'] += 1
                t = state['t']
                state['m'].mul_(b1).add_(g, alpha=1 - b1)
                state['v'].mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = state['m'] / (1 - b1 ** t)
                v_hat = state['v'] / (1 - b2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    """AdamW = Adam con weight decay desacoplado (mismo default que torch.optim.AdamW)."""
    return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=True)
