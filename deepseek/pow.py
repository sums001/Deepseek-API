"""
Proof-of-work solver for DeepSeek's chat completion endpoint.

DeepSeek gates `POST /api/v0/chat/completion` behind a proof-of-work header
(`x-ds-pow-response`). The PoW algorithm ("DeepSeekHashV1") is shipped as a
WebAssembly module the website loads from its own CDN
(fe-static.deepseek.com/.../sha3_wasm_bg.wasm). Rather than reimplement its
exact float64 hashing logic, we run DeepSeek's own module — the same code the
browser runs — inside the `wasmtime` sandbox (no file/network access).

Public API:
    solver = DeepSeekPow()                      # loads the wasm once
    header = solver.make_header(challenge_dict)  # -> base64 x-ds-pow-response value

`challenge_dict` is the `biz_data.challenge` object returned by
`POST /api/v0/chat/create_pow_challenge`.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Optional

import wasmtime

WASM_PATH = Path(__file__).resolve().parent / "sha3_wasm_bg.wasm"


_ENGINE = wasmtime.Engine()
_MODULE = wasmtime.Module.from_file(_ENGINE, str(WASM_PATH))


class DeepSeekPow:
    def __init__(self, wasm_path: Path = WASM_PATH):
        if wasm_path == WASM_PATH:
            self._engine = _ENGINE
            self._module = _MODULE
        else:
            self._engine = wasmtime.Engine()
            self._module = wasmtime.Module.from_file(self._engine, str(wasm_path))

    def solve(self, challenge: str, prefix: str, difficulty: float) -> Optional[int]:
        """Return the integer PoW answer, or None if the module reports failure.

        Mirrors the website's wasm-bindgen call:
            wasm_solve(retptr, challenge_ptr, challenge_len,
                       prefix_ptr, prefix_len, difficulty)
        with a 16-byte return slot reserved on the shadow stack. The slot holds
        an i32 status flag at +0 and an f64 answer at +8.
        """
        store = wasmtime.Store(self._engine)
        inst = wasmtime.Instance(store, self._module, [])
        exp = inst.exports(store)
        memory: wasmtime.Memory = exp["memory"]
        solve_func = exp["wasm_solve"]
        malloc_func = exp["__wbindgen_export_0"]            # malloc(size, align)
        add_to_stack_func = exp["__wbindgen_add_to_stack_pointer"]

        def write_str(text: str) -> tuple[int, int]:
            """malloc + copy a UTF-8 string into wasm memory; return (ptr, len)."""
            data = text.encode("utf-8")
            ptr = malloc_func(store, len(data), 1)
            base = memory.data_ptr(store)
            for i, b in enumerate(data):
                base[ptr + i] = b
            return ptr, len(data)

        retptr = add_to_stack_func(store, -16)
        try:
            c_ptr, c_len = write_str(challenge)
            p_ptr, p_len = write_str(prefix)
            solve_func(store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty))

            mem = memory.data_ptr(store)
            status = struct.unpack("<i", bytes(mem[retptr:retptr + 4]))[0]
            value = struct.unpack("<d", bytes(mem[retptr + 8:retptr + 16]))[0]
        finally:
            add_to_stack_func(store, 16)

        if status == 0:
            return None
        return int(value)

    def make_header(self, challenge: dict) -> str:
        """Build the base64 `x-ds-pow-response` header value from a challenge dict."""
        prefix = f"{challenge['salt']}_{challenge['expire_at']}_"
        answer = self.solve(challenge["challenge"], prefix, challenge["difficulty"])
        if answer is None:
            raise RuntimeError("PoW solver returned no answer (challenge expired?)")
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("utf-8")
