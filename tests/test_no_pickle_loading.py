import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# tools/convert_checkpoints.py must unpickle by definition; it is deleted
# once the committed checkpoints are converted.
ALLOWED = {os.path.join("tools", "convert_checkpoints.py")}


def python_sources():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "docs"}]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                yield os.path.relpath(path, REPO_ROOT), path


def unexempt_sources():
    allowed = {a.replace("\\", "/") for a in ALLOWED}
    for relative, path in python_sources():
        if relative.replace("\\", "/") not in allowed:
            yield relative, path


def test_no_source_file_calls_torch_load():
    offenders = []
    for relative, path in unexempt_sources():
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if re.search(r"\btorch\.load\s*\(", line):
                    offenders.append("%s:%d" % (relative, number))
    assert offenders == [], "torch.load found at: {}".format(offenders)


def test_no_pt_checkpoints_remain():
    models_dir = os.path.join(REPO_ROOT, "vae_motion", "models")
    leftovers = [n for n in os.listdir(models_dir) if n.endswith(".pt")]
    assert leftovers == []


def test_no_source_file_globs_for_pt_files():
    offenders = []
    for relative, path in unexempt_sources():
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if re.search(r"""["'][^"']*\*\.pt["']""", line):
                    offenders.append("%s:%d" % (relative, number))
    assert offenders == [], "glob for *.pt found at: {}".format(offenders)
