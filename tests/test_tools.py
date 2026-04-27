import json

import pytest

from nanovlm.train.tools.browser import extract_text
from nanovlm.train.tools.filesystem import FileSystemTool
from nanovlm.train.tools.python import PythonSandbox


def test_python_sandbox_runs_simple_program():
    sb = PythonSandbox(timeout_s=15)
    out = sb({"code": "print(40 + 2)"})
    assert out["returncode"] == 0
    assert "42" in out["stdout"]


def test_python_sandbox_times_out():
    sb = PythonSandbox(timeout_s=2)
    out = sb({"code": "import time\ntime.sleep(10)\nprint('done')"})
    assert out["timed_out"] is True


def test_python_sandbox_truncates_output():
    sb = PythonSandbox(timeout_s=15, max_output_chars=64)
    out = sb({"code": "print('A' * 1000)"})
    assert len(out["stdout"]) <= 64


def test_filesystem_write_and_read_roundtrip(tmp_path):
    fs = FileSystemTool(root=tmp_path)
    w = fs({"op": "write", "path": "hello.txt", "content": "hi there"})
    assert w["ok"] is True
    r = fs({"op": "read", "path": "hello.txt"})
    assert r["ok"] and r["text"] == "hi there"


def test_filesystem_rejects_path_escape(tmp_path):
    fs = FileSystemTool(root=tmp_path)
    with pytest.raises(ValueError):
        fs({"op": "read", "path": "../../etc/passwd"})


def test_filesystem_grep(tmp_path):
    fs = FileSystemTool(root=tmp_path)
    fs({"op": "write", "path": "a.txt", "content": "foo bar\nbaz qux\n"})
    fs({"op": "write", "path": "b.txt", "content": "no match here\n"})
    out = fs({"op": "grep", "path": "", "pattern": "bar"})
    assert out["ok"] and len(out["hits"]) == 1
    assert out["hits"][0]["text"].strip() == "foo bar"


def test_browser_extracts_visible_text():
    html = "<html><head><script>blocked</script></head><body><p>Hello <b>world</b>.</p><div>Done.</div></body></html>"
    text = extract_text(html)
    assert "Hello world." in text
    assert "Done." in text
    assert "blocked" not in text


def test_image_op_crop(tmp_path):
    from PIL import Image

    from nanovlm.train.tools.image_ops import image_op

    src = tmp_path / "src.png"
    Image.new("RGB", (100, 80), "red").save(src)
    out = image_op("crop", path=str(src), bbox=[10, 20, 60, 70])
    assert out["ok"] is True
    assert out["width"] == 50 and out["height"] == 50
