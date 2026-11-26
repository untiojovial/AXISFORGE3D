"""Phase VII representation experiment: small MLPs over candidate, geometry and frozen visual features.

Feature blocks (PHASE7_PROTOCOL.md, I.3):
    v  candidate and robot features (16): the Phase V features without the object extents;
    u  object geometry (35): log bounding-box extents + D2 shape distribution;
    d  object visual embedding (global, frozen): one vector per part.
Every model sees v; u and d are optional branches. A 'linear' visual branch is a single linear map.
"""
from __future__ import annotations

import time

import numpy as np

from research_grasping.phase5 import baselines as bl

V_COLUMNS = [i for i, f in enumerate(bl.FEATURE_NAMES) if not f.startswith('log_extent')]
EXTENT_COLUMNS = [i for i, f in enumerate(bl.FEATURE_NAMES) if f.startswith('log_extent')]
LATENT = 16
GRASP_HIDDEN, GRASP_OUT, GEOM_HIDDEN, VIS_HIDDEN, HEAD_HIDDEN = 64, 32, 32, 64, 64
DROPOUT, LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE = .1, 1e-3, 1e-4, 4096, 40, 5


def name_groups(names, threshold=.5):
    """Union of parts whose name-word Jaccard index is >= threshold (the Phase V variant rule)."""
    words = {n: set(n.lower().replace('-', '_').split('_')) - {''} for n in names}
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if len(words[a] | words[b]) and len(words[a] & words[b])/len(words[a] | words[b]) >= threshold:
                parent[find(a)] = find(b)
    return {n: find(n) for n in names}


def grouped_folds(names, groups, k=5, seed=20261201):
    keys = sorted(set(groups.values()))
    order = np.random.default_rng(seed).permutation(len(keys))
    fold_of = {keys[i]: j % k for j, i in enumerate(order)}
    return [[n for n in names if fold_of[groups[n]] == f] for f in range(k)]


class Standardiser:
    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-8
        return self

    def __call__(self, x):
        return (x - self.mean)/self.std


def build_net(dims, geometry, visual, linear_visual, latent=LATENT):
    import torch
    from torch import nn

    def mlp(i, h, o):
        return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(h, o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.grasp = mlp(dims['v'], GRASP_HIDDEN, GRASP_OUT)
            width = GRASP_OUT
            self.geometry = mlp(dims['u'], GEOM_HIDDEN, latent) if geometry else None
            width += latent if geometry else 0
            if visual:
                self.visual = nn.Linear(dims['d'], latent) if linear_visual else mlp(dims['d'], VIS_HIDDEN, latent)
                width += latent
            else:
                self.visual = None
            self.head = nn.Sequential(nn.Linear(width, HEAD_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HEAD_HIDDEN, 1))

        def forward(self, v, u=None, d=None):
            z = [self.grasp(v)]
            if self.geometry is not None:
                z.append(self.geometry(u))
            if self.visual is not None:
                z.append(self.visual(d))
            return self.head(torch.cat(z, 1)).squeeze(1)
    return Net()


class MLPModel:
    """Fit/predict wrapper. data: dict part -> dict(v=(n,16), y=(n,), u=(35,), d=(D,))."""

    def __init__(self, geometry=False, visual=False, linear_visual=False, loss='bce', latent=LATENT, seed=0):
        self.geometry, self.visual, self.linear_visual = geometry, visual, linear_visual
        self.loss, self.latent, self.seed = loss, latent, seed

    def _stack(self, data, parts):
        v = np.vstack([data[p]['v'] for p in parts]).astype(np.float32)
        y = np.concatenate([data[p]['y'] for p in parts]).astype(np.float32)
        idx = np.concatenate([np.full(len(data[p]['y']), i) for i, p in enumerate(parts)])
        return v, y, idx

    def fit(self, data, train_parts, val_parts):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(1)
        start = time.perf_counter()
        vt, yt, it = self._stack(data, train_parts)
        self.sv = Standardiser().fit(vt)
        if self.geometry:
            self.su = Standardiser().fit(np.vstack([data[p]['u'] for p in train_parts]))
        if self.visual:
            self.sd = Standardiser().fit(np.vstack([data[p]['d'] for p in train_parts]))
        dims = dict(v=vt.shape[1], u=len(data[train_parts[0]]['u']), d=len(data[train_parts[0]]['d']) if self.visual else 0)
        self.net = build_net(dims, self.geometry, self.visual, self.linear_visual, self.latent)
        opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        T = self._tensors(data, train_parts)
        V = self._tensors(data, val_parts)
        best, best_state, bad, self.epochs = np.inf, None, 0, 0
        rng = np.random.default_rng(self.seed)
        for epoch in range(MAX_EPOCHS):
            self.net.train()
            for batch in self._batches(T, rng):
                opt.zero_grad()
                loss = self._loss(T, batch)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(sum(self._loss(V, b).item()*len(b[0]) for b in self._batches(V, None))/max(len(V['y']), 1))
            self.epochs = epoch+1
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: x.clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.train_seconds = time.perf_counter() - start
        self.params = int(sum(p.numel() for p in self.net.parameters()))
        return self

    def _tensors(self, data, parts):
        import torch
        v, y, idx = self._stack(data, parts)
        out = dict(v=torch.tensor(self.sv(v)), y=torch.tensor(y), idx=torch.tensor(idx), parts=parts,
                   offsets=np.cumsum([0]+[len(data[p]['y']) for p in parts]))
        if self.geometry:
            out['u'] = torch.tensor(self.su(np.vstack([data[p]['u'] for p in parts])).astype(np.float32))
        if self.visual:
            out['d'] = torch.tensor(self.sd(np.vstack([data[p]['d'] for p in parts])).astype(np.float32))
        return out

    def _batches(self, T, rng):
        import torch
        if self.loss == 'bce':
            n = len(T['y'])
            order = rng.permutation(n) if rng is not None else np.arange(n)
            for s in range(0, n, BATCH):
                yield (torch.tensor(order[s:s+BATCH]),)
        else:  # listwise: groups of whole parts
            P = len(T['parts'])
            order = rng.permutation(P) if rng is not None else np.arange(P)
            for s in range(0, P, 16):
                ps = order[s:s+16]
                rows = np.concatenate([np.arange(T['offsets'][p], T['offsets'][p+1]) for p in ps])
                yield (torch.tensor(rows), ps)

    def _forward(self, T, rows):
        u = T['u'][T['idx'][rows]] if self.geometry else None
        d = T['d'][T['idx'][rows]] if self.visual else None
        return self.net(T['v'][rows], u, d)

    def _loss(self, T, batch):
        import torch
        import torch.nn.functional as F
        rows = batch[0]
        logits = self._forward(T, rows)
        y = T['y'][rows]
        if self.loss == 'bce':
            return F.binary_cross_entropy_with_logits(logits, y)
        total, count, pos = 0., 0, 0
        for p in batch[1]:
            sel = (T['idx'][rows] == int(p))
            lp, yp = logits[sel], y[sel]
            if yp.sum() > 0:
                total = total - (torch.log_softmax(lp, 0)*(yp/yp.sum())).sum()
                pos += 1
            else:
                total = total + .1*F.binary_cross_entropy_with_logits(lp, yp)*1.
            count += 1
        return total/max(count, 1)

    def predict(self, data, parts):
        import torch
        T = self._tensors(data, parts)
        self.net.eval()
        out = {}
        with torch.no_grad():
            start = time.perf_counter()
            logits = self._forward(T, torch.arange(len(T['y'])))
            self.predict_seconds = time.perf_counter() - start
        p = torch.sigmoid(logits).numpy()
        for i, part in enumerate(parts):
            out[part] = p[T['offsets'][i]:T['offsets'][i+1]]
        return out
