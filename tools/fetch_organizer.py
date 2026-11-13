"""Fetch the permitted organizer snapshot and verify every Git blob hash."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "739d627b20fdb4899ab7487723d2e4fccec27033"
REPOSITORY = "JuliusRye/QEC_Decoder_Hakathon_2026"


def main():
    destination = ROOT / "organizer"
    manifest = destination / "SOURCE_MANIFEST.json"
    if manifest.exists():
        raise FileExistsError("Snapshot already exists; verify it instead of downloading again.")
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
    with urlopen(url, timeout=30) as response:
        tree = json.load(response)
    assert not tree.get("truncated"), "Incomplete repository tree"
    selected = [
        entry for entry in tree["tree"]
        if entry["type"] == "blob" and (
            entry["path"].startswith("src/hackathon_2026/")
            or entry["path"].startswith("submission_template/")
            or entry["path"] in {
                ".python-version", "LICENSE", "README.md", "pyproject.toml",
                "notebooks/getting_started.ipynb",
            }
        )
    ]
    records = {}
    for entry in selected:
        path = destination / entry["path"]
        source = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{entry['path']}"
        if path.exists():
            content = path.read_bytes()
        else:
            with urlopen(source, timeout=30) as response:
                content = response.read()
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        if hashlib.sha1(blob).hexdigest() != entry["sha"]:
            raise ValueError(f"Git blob hash mismatch: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        records[entry["path"]] = {
            "git_blob": entry["sha"], "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content), "url": source,
        }
        print(f"Verified {entry['path']}", flush=True)
    with manifest.open("x") as stream:
        json.dump({"repository": REPOSITORY, "revision": REVISION, "files": records}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
