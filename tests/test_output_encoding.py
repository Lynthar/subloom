"""Output that is not a console: a pipe or file in a legacy code page must not crash the CLI."""

import os
import subprocess
import sys


def test_cjk_output_to_a_cp1252_pipe_succeeds_as_utf8(tmp_path):
    log_file = tmp_path / "字幕.log"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    args = ["--config", str(tmp_path / "config.yaml"), "config", "set", "log_file", str(log_file)]

    result = subprocess.run([sys.executable, "-m", "subloom", *args], capture_output=True, env=env)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "字幕" in result.stdout.decode("utf-8")
