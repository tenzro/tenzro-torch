# tenzro-torch

PyTorch helpers for [Tenzro](https://www.tenzro.com). Small on purpose; each piece says what has actually been tested.

```bash
pip install "git+https://github.com/tenzro/tenzro-torch[onnx]"
```

## Remote modules (`RemoteEmbedding`, `RemoteChat`)

`nn.Module`s whose `forward` calls a Tenzro node's OpenAI-shaped routes (`/v1/embeddings`, `/v1/chat/completions`) instead of running weights locally. They hold no parameters, return no gradients, and need only `httpx`.

```python
from tenzro_torch import RemoteEmbedding
emb = RemoteEmbedding("http://127.0.0.1:8545/v1", "<embedding model id>")
vectors = emb(["hello", "world"])   # torch.float32, shape (2, dim)
```

**Status: mechanics tested, not a successful round trip.** Unit tests run against a local stub that speaks the OpenAI wire shape. Against the live dev node (chain 13380) the routes answered HTTP 404 `model_not_served`, because that node had no model loaded, and `TenzroApiError` surfaced that correctly. A real completion or embedding from a Tenzro node has **not** been observed. The node's payment gate (HTTP 402) is not handled.

## `export_onnx`

```python
from tenzro_torch import export_onnx
max_abs_diff = export_onnx(model, (example_input,), "model.onnx")
```

Calls `torch.onnx.export` (TorchScript exporter, `dynamo=False`), checks the file with `onnx.checker`, runs it with onnxruntime, and raises `ExportMismatch` if outputs differ from the original. Single-tensor-output models only. Tested on a 2-layer MLP (including a different batch size than the export example). Not tested on real Tenzro models, convolutional nets or transformers.

## Not done / untested

- Publishing to the model catalog: the CLI has `tenzro catalog publish <file|dir> --key-label <label>` (signs with a TPM/Secure Enclave key). Found in the source, **not run**, no wrapper here.
- Training-job submission: the node has `tenzro_loadTrainingModel` and training-round commit/settle RPCs. **Not tried**, no wrapper here.
- Streaming chat, tool calls (see the `tenzro` SDK's `ChatClient` for the streaming contract).

## Development

```bash
pip install -e ".[test]" && pytest
```
CI runs the same, offline.
