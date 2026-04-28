"""Minimal stdlib HTTP chat UI. No Flask, no React, no JS frameworks.

Serves a single HTML page that POSTs ``{messages}`` JSON to ``/chat`` and
appends the streamed reply. Designed for a quick local demo.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from nanovlm.models.qwen3_5 import DEFAULT_MODEL_PATH, render_chat
from nanovlm.train.common import init_runtime
from nanovlm.train.engine import generate
from nanovlm.train.model_factory import build_model, load_tokenizer

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>NanoVLM chat</title>
<style>body{font-family:system-ui,sans-serif;max-width:780px;margin:2em auto;padding:0 1em}
.msg{margin:1em 0;padding:0.6em 0.9em;border-radius:8px;white-space:pre-wrap}
.user{background:#f0f3ff}
.assistant{background:#fff;border:1px solid #ddd}
form{display:flex;gap:.5em}
textarea{flex:1;min-height:3em;padding:.5em;font-family:inherit}
button{padding:.5em 1em}</style></head><body>
<h2>NanoVLM chat</h2>
<div id="log"></div>
<form id="f"><textarea id="t" placeholder="ask…"></textarea><button>Send</button></form>
<script>
const log=document.getElementById('log'),t=document.getElementById('t'),f=document.getElementById('f');
let history=[];
function add(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;log.appendChild(d);d.scrollIntoView();}
f.onsubmit=async(e)=>{e.preventDefault();const v=t.value.trim();if(!v)return;t.value='';add('user',v);history.push({role:'user',content:v});
const r=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages:history})});
const j=await r.json();add('assistant',j.reply);history.push({role:'assistant',content:j.reply});};
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    ctx = None
    lock = threading.Lock()
    max_new_tokens = 512
    temperature = 0.7
    top_p = 0.95

    def log_message(self, format, *args):  # silence default access logs
        return

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, "text/plain", b"not found")
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        messages = body.get("messages") or []
        with self.__class__.lock:
            prompt = render_chat(messages, add_generation_prompt=True)
            ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.ctx.device)
            out = generate(
                self.model, ids,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                temperature=self.temperature,
                top_p=self.top_p,
                use_cache=True,
            )
            reply = self.tokenizer.decode(out.sequences[0, ids.size(1):].tolist()).rstrip()
            if reply.endswith("<|im_end|>"):
                reply = reply[: -len("<|im_end|>")].rstrip()
        self._send(200, "application/json", json.dumps({"reply": reply}).encode("utf-8"))

    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny stdlib chat UI for Qwen3.5.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--init", choices=["scratch", "checkpoint"], default="checkpoint")
    p.add_argument("--device-type", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    ctx = init_runtime(args.device_type, args.dtype)
    tokenizer = load_tokenizer(args.model_path)
    model = build_model(model_path=args.model_path, init=args.init, device=ctx.device, dtype=ctx.dtype, tiny=args.tiny, text_only=args.text_only).eval()

    _Handler.model = model
    _Handler.tokenizer = tokenizer
    _Handler.ctx = ctx
    _Handler.max_new_tokens = args.max_new_tokens

    srv = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"chat UI: http://127.0.0.1:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
