"""torch -> ONNX export with a numerical check against onnxruntime."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ExportMismatch(AssertionError):
    """The exported graph disagrees with the original module beyond tolerance."""


def export_onnx(model: nn.Module, example_inputs: tuple[torch.Tensor, ...], path: str, *,
                rtol: float = 1e-4, atol: float = 1e-5, dynamic_batch: bool = True, opset_version: int | None = None) -> float:
    """Export ``model`` with ``torch.onnx.export``, run both on ``example_inputs`` and compare.

    Returns the max absolute difference; raises :class:`ExportMismatch` if outside tolerance.
    ``dynamic_batch`` marks dim 0 of every input and output dynamic.
    """
    import onnx
    import onnxruntime as ort

    model = model.eval()
    n_in = len(example_inputs)
    in_names = [f"input{i}" for i in range(n_in)]
    kwargs = {}
    if opset_version is not None:
        kwargs["opset_version"] = opset_version
    if dynamic_batch:
        kwargs["dynamic_axes"] = {n: {0: "batch"} for n in in_names + ["output"]}
    with torch.no_grad():
        expected = model(*example_inputs)
        if not isinstance(expected, torch.Tensor):
            raise TypeError("export_onnx supports models returning a single tensor")
        torch.onnx.export(model, example_inputs, path, input_names=in_names, output_names=["output"],
                          dynamo=False, **kwargs)
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: x.detach().numpy() for n, x in zip(in_names, example_inputs)})[0]
    want = expected.detach().numpy()
    if got.shape != want.shape or not np.allclose(got, want, rtol=rtol, atol=atol):
        raise ExportMismatch(f"max abs diff {np.abs(got - want).max() if got.shape == want.shape else 'shape mismatch'}")
    return float(np.abs(got - want).max())
