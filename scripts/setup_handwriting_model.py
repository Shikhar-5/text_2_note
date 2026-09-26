"""Download the published HWT inference code and checkpoint once."""

from pathlib import Path

from huggingface_hub import snapshot_download


root = Path(__file__).resolve().parents[1]
target = root / "Text2Note" / "handwriting_engine" / "hwt_model"
target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="ankankbhunia/HWT",
    repo_type="space",
    local_dir=target,
    allow_patterns=[
        "params.py",
        "models/*.py",
        "models/sync_batchnorm/*.py",
        "data/dataset.py",
        "util/*.py",
        "files/iam_model.pth",
    ],
)

checkpoint = target / "files" / "iam_model.pth"
if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000_000:
    raise SystemExit("The HWT checkpoint did not download completely.")
print(f"Personal handwriting model ready: {target}")
