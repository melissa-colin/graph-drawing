"""Carbon tracker: one CSV row per block, even when the block or the script fails."""
import pytest

from tools.carbon import markdown_table, read_runs, track
from tools.run_tracked import main as run_tracked_main

FAST = dict(measure_power_secs=1)


def test_track_writes_one_row(tmp_path):
    out = tmp_path / "runs" / "emissions.csv"  # missing folder: created
    with track("unit", out, **FAST):
        sum(i * i for i in range(10_000))
    with track("unit-2", out, **FAST):
        pass
    runs = read_runs(out)
    assert [r["name"] for r in runs] == ["unit", "unit-2"]
    for r in runs:
        assert r["duration_s"] >= 0 and r["energy_kwh"] >= 0 and r["co2_g"] >= 0


def test_track_writes_the_row_when_the_block_raises(tmp_path):
    out = tmp_path / "emissions.csv"
    with pytest.raises(ZeroDivisionError):
        with track("fails", out, **FAST):
            1 / 0
    assert [r["name"] for r in read_runs(out)] == ["fails"]


def test_markdown_table_totals():
    runs = [dict(name="a", duration_s=30, energy_kwh=0.001, co2_g=0.05, cpu="cpu", gpu=""),
            dict(name="b", duration_s=3700, energy_kwh=0.002, co2_g=0.10, cpu="cpu", gpu="RTX")]
    table = markdown_table(runs).splitlines()
    assert table[2] == "| a | cpu | 0 min 30 s | 0.0010 | 0.05 |"
    assert table[3] == "| b | RTX | 1 h 01 min | 0.0020 | 0.10 |"
    assert table[4] == "| **Total** | | 1 h 02 min | 0.0030 | 0.15 |"


def _script(tmp_path, body):
    folder = tmp_path / "scripts"
    folder.mkdir()
    (folder / "helper.py").write_text("VALUE = 42\n")
    script = folder / "job.py"
    script.write_text(body)
    return script


def test_run_tracked_runs_the_script_like_python_does(tmp_path):
    marker = tmp_path / "marker.txt"
    # imports a module of its own folder, reads its arguments, checks __name__
    script = _script(tmp_path, "import sys\nfrom helper import VALUE\n"
                               "assert __name__ == '__main__'\n"
                               "open(sys.argv[1], 'w').write(f'{VALUE} {sys.argv[2]}')\n")
    out = tmp_path / "emissions.csv"
    run_tracked_main(["--name", "job", "--out", str(out), str(script), str(marker), "hello"])
    assert marker.read_text() == "42 hello"
    assert [r["name"] for r in read_runs(out)] == ["job"]


def test_run_tracked_keeps_the_exit_code_and_the_row(tmp_path):
    script = _script(tmp_path, "import sys\nsys.exit(3)\n")
    out = tmp_path / "emissions.csv"
    with pytest.raises(SystemExit) as exit_info:
        run_tracked_main(["--out", str(out), str(script)])
    assert exit_info.value.code == 3
    assert [r["name"] for r in read_runs(out)] == ["job"]  # default name: the script name
