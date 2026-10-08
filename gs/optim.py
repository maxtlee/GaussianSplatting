"""A minimal Adam optimizer with resizable per-Gaussian state.

Densification changes the number of Gaussians, so the optimizer moments must be
resized alongside the parameters. ``torch.optim.Adam`` does not support that
cleanly, so this implementation keeps the moments as plain leading-dimension
tensors that can be indexed, concatenated, and zeroed.
"""

import torch


class AdamStore:
    """Adam over a dict of same-leading-dimension parameter tensors.

    Parameters
    ----------
    params : dict[str, Tensor]
        Leaf tensors updated in place. All are indexed by the same Gaussian
        axis.
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}

    def zero_grad(self, params):
        for p in params.values():
            p.grad = None

    def step(self, params):
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        with torch.no_grad():
            for k, p in params.items():
                g = p.grad
                if g is None:
                    continue
                m, v = self.m[k], self.v[k]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                denom = (v / bc2).sqrt().add_(self.eps)
                p.addcdiv_(m / bc1, denom, value=-self.lr)

    def resize(self, old_index, is_new):
        """Resize moments after densification.

        ``old_index`` maps each new Gaussian to a row of the old state (only
        meaningful where ``is_new`` is ``False``); moments for new Gaussians are
        zeroed.
        """
        for k in self.m:
            m = self.m[k][old_index].clone()
            v = self.v[k][old_index].clone()
            m[is_new] = 0.0
            v[is_new] = 0.0
            self.m[k] = m
            self.v[k] = v
