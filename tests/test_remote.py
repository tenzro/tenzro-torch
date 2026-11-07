"""Mechanics only: a local stub that speaks the OpenAI wire shape. This does NOT test a real Tenzro node."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch

from tenzro_torch import RemoteChat, RemoteEmbedding, TenzroApiError


class H(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        H.seen.append((self.path, body, self.headers.get("Authorization")))
        if body["model"] == "missing":
            code, out = 404, {"error": {"message": "model missing not served here", "type": "invalid_request_error", "code": "model_not_served"}}
        elif self.path == "/v1/embeddings":
            inp = [body["input"]] if isinstance(body["input"], str) else body["input"]
            code, out = 200, {"object": "list", "data": [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in reversed(list(enumerate(inp)))]}
        else:
            code, out = 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        raw = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def base():
    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    H.seen.clear()
    yield f"http://127.0.0.1:{s.server_port}/v1"
    s.shutdown()


def test_embedding_tensor_order_and_float_encoding(base):
    out = RemoteEmbedding(base, "e", api_key="k")(["a", "bbb"])
    assert out.shape == (2, 3) and out.dtype == torch.float32
    assert out[:, 0].tolist() == [1.0, 3.0]
    path, body, auth = H.seen[0]
    assert path == "/v1/embeddings" and body["encoding_format"] == "float" and auth == "Bearer k"


def test_chat(base):
    assert RemoteChat(base, "c")("hi") == "ok"
    assert H.seen[0][1]["messages"] == [{"role": "user", "content": "hi"}]


def test_node_error_shape(base):
    with pytest.raises(TenzroApiError) as e:
        RemoteChat(base, "missing")("hi")
    assert e.value.status == 404 and e.value.code == "model_not_served"
