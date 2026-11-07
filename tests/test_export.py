import pytest
import torch
from torch import nn

from tenzro_torch import ExportMismatch, export_onnx


def toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_export_matches_original(tmp_path):
    x = torch.randn(3, 4)
    diff = export_onnx(toy(), (x,), str(tmp_path / "m.onnx"))
    assert diff < 1e-5


def test_dynamic_batch(tmp_path):
    import onnxruntime as ort
    m = toy()
    p = str(tmp_path / "m.onnx")
    export_onnx(m, (torch.randn(2, 4),), p)
    y = torch.randn(7, 4)
    got = ort.InferenceSession(p).run(None, {"input0": y.numpy()})[0]
    assert abs(got - m(y).detach().numpy()).max() < 1e-5


def test_mismatch_is_detected(tmp_path):
    class Drift(nn.Module):  # output differs between eager and traced graph
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 2)
        def forward(self, x):
            return self.l(x) if not torch.jit.is_tracing() and not torch.compiler.is_exporting() else self.l(x) + 1.0
    with pytest.raises(ExportMismatch):
        export_onnx(Drift(), (torch.randn(2, 4),), str(tmp_path / "d.onnx"))
