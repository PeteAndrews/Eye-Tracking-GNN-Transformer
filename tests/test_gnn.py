"""M4 unit tests: shapes, gradients, attention, panel probe distinguishability."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.gnn import CompactGNN, PanelProbe, mask_panel_features, panel_labels_from_x

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data_processed" / "graphs" / "g1_bge1024" / "T01__not_eligible.pt"


def _toy_batch(n: int = 6, e: int = 10, in_dim: int = 32, n_rel: int = 5):
    torch.manual_seed(0)
    x = torch.randn(n, in_dim)
    # Put panel one-hot in first 6 dims for toy (text_dim=0)
    x[:, :6] = 0
    for i in range(n):
        x[i, i % 6] = 1.0
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, n_rel, (e,))
    edge_attr = torch.randn(e, 5)
    return x, edge_index, edge_type, edge_attr


def test_compact_gnn_shapes_and_grad():
    x, ei, et, ea = _toy_batch()
    model = CompactGNN(
        in_dim=32,
        hidden_dim=16,
        out_dim=16,
        n_layers=2,
        n_heads=4,
        n_relations=5,
        relation_emb_dim=8,
        edge_attr_dim=5,
    )
    x_v, h_v = model(x, ei, et, ea)
    assert x_v.shape == (6, 16)
    assert h_v.shape == (6, 16)
    loss = h_v.sum()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_attention_extractable():
    x, ei, et, ea = _toy_batch(e=8)
    model = CompactGNN(in_dim=32, hidden_dim=16, out_dim=16, n_heads=4, relation_emb_dim=8)
    model.eval()
    _, h_v = model(x, ei, et, ea, return_attention=True)
    attn = model.last_attention
    assert attn is not None
    assert attn.shape[0] == ei.size(1)
    assert torch.isfinite(attn).all()


def test_empty_edges():
    x = torch.randn(4, 32)
    ei = torch.zeros(2, 0, dtype=torch.long)
    et = torch.zeros(0, dtype=torch.long)
    ea = torch.zeros(0, 5)
    model = CompactGNN(in_dim=32, hidden_dim=16, out_dim=16, n_heads=4, relation_emb_dim=8)
    x_v, h_v = model(x, ei, et, ea)
    assert x_v.shape[0] == 4 and h_v.shape[0] == 4


@pytest.mark.skipif(not GRAPH.is_file(), reason="M3 graph not built")
def test_real_graph_forward():
    g = torch.load(GRAPH, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(ROOT / "configs" / "gnn.yaml")
    model = CompactGNN(
        in_dim=int(cfg.model.in_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        out_dim=int(cfg.model.out_dim),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        n_relations=int(cfg.model.n_relations),
        relation_emb_dim=int(cfg.model.relation_emb_dim),
        edge_attr_dim=int(cfg.model.edge_attr_dim),
        dropout=float(cfg.model.dropout),
        edge_dropout=float(cfg.model.edge_dropout),
    )
    model.eval()
    x_v, h_v = model(g["x"], g["edge_index"], g["edge_type"], g["edge_attr"], return_attention=True)
    assert x_v.shape[0] == g["x"].shape[0]
    assert h_v.shape[-1] == int(cfg.model.out_dim)
    assert model.last_attention is not None
    assert model.last_attention.shape[0] == g["edge_index"].size(1)


@pytest.mark.skipif(not GRAPH.is_file(), reason="M3 graph not built")
def test_panel_probe_h_v_beats_featureless_baseline():
    """With panel features masked, h_v should recover panel better than chance via MP."""
    g = torch.load(GRAPH, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(ROOT / "configs" / "gnn.yaml")
    text_dim = int(cfg.model.text_dim)
    n_panel = int(cfg.model.panel_vocab_size)
    x = mask_panel_features(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
    # Labels from *unmasked* original
    y = panel_labels_from_x(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
    # Only segment nodes (exclude abstract panels for a cleaner probe)
    n_seg = int(g["n_segments"])
    y = y[:n_seg]

    torch.manual_seed(13)
    model = CompactGNN(
        in_dim=int(cfg.model.in_dim),
        hidden_dim=64,
        out_dim=64,
        n_layers=2,
        n_heads=4,
        n_relations=5,
        relation_emb_dim=16,
        edge_attr_dim=5,
        dropout=0.0,
        edge_dropout=0.0,
    )
    probe = PanelProbe(64, n_classes=n_panel)
    opt = torch.optim.Adam(list(model.parameters()) + list(probe.parameters()), lr=1e-2)

    model.train()
    probe.train()
    for _ in range(60):
        opt.zero_grad()
        _, h = model(x, g["edge_index"], g["edge_type"], g["edge_attr"])
        logits = probe(h[:n_seg])
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

    model.eval()
    probe.eval()
    with torch.no_grad():
        _, h = model(x, g["edge_index"], g["edge_type"], g["edge_attr"])
        pred = probe(h[:n_seg]).argmax(-1)
        acc = (pred == y).float().mean().item()
    # Better than uniform chance (1/6≈0.17); on a tiny graph expect clearly above chance
    assert acc >= 0.5, f"panel probe acc={acc:.3f} too low — message passing may be broken"


def test_stable_across_three_seeds():
    accs = []
    for seed in (13, 42, 1337):
        torch.manual_seed(seed)
        x, ei, et, ea = _toy_batch(n=8, e=16, in_dim=24)
        # panel labels in cols 0:6
        y = x[:, :6].argmax(-1)
        x_masked = x.clone()
        x_masked[:, :6] = 0
        model = CompactGNN(in_dim=24, hidden_dim=16, out_dim=16, n_heads=4, relation_emb_dim=8, dropout=0.0, edge_dropout=0.0)
        probe = PanelProbe(16, 6)
        opt = torch.optim.Adam(list(model.parameters()) + list(probe.parameters()), lr=2e-2)
        for _ in range(40):
            opt.zero_grad()
            _, h = model(x_masked, ei, et, ea)
            loss = F.cross_entropy(probe(h), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, h = model(x_masked, ei, et, ea)
            acc = (probe(h).argmax(-1) == y).float().mean().item()
        accs.append(acc)
    assert all(a >= 0.4 for a in accs), accs
    # Not wildly unstable
    assert max(accs) - min(accs) < 0.6
