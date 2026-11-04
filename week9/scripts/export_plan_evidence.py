"""Export or check the reviewed contraction-plan evidence, without raw model data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "evidence" / "plans"
CASES = {
    "asia-ad-random": ("runs/random", "random", 42, []),
    "asia-ad-greedy": ("runs/greedy", "greedy", None, []),
    "promedus-greedy": ("artifacts/promedus-greedy", "greedy", None, [59, 102, 139]),
    "promedus-treesa": ("artifacts/promedus-treesa", "treesa", 42, [59, 102, 139]),
}


def check_plan(plan, method, seed, query):
    assert plan["method"] == method and plan["seed"] == seed and plan["query"] == query
    expected_keys = {"method", "seed", "query", "input_scopes", "steps",
                     "total_work", "largest_tensor", "largest_generated", "settings"}
    assert set(plan) == expected_keys, "unexpected plan content"
    scopes = plan["input_scopes"]
    live = {i: set(scope) for i, scope in enumerate(scopes)}
    largest = max([2 ** len(scope) for scope in live.values()] + [1])
    generated = 1
    total = 0
    for index, step in enumerate(plan["steps"]):
        assert set(step) == {"inputs", "output", "input_labels", "retained_labels", "work", "output_entries"}
        assert step["output"] == len(scopes) + index
        inputs = [live.pop(i) for i in step["inputs"]]
        assert len(inputs) in (1, 2)
        assert [set(x) for x in step["input_labels"]] == inputs
        union = set().union(*inputs)
        keep = union.intersection(set(query).union(*live.values()))
        assert keep == set(step["retained_labels"])
        work = 2 ** len(union)
        size = 2 ** len(keep)
        assert step["work"] == work and step["output_entries"] == size
        total += work
        largest = max(largest, size)
        generated = max(generated, size)
        live[step["output"]] = keep
    assert len(live) == 1 and next(iter(live.values())) == set(query)
    assert (total, largest, generated) == (plan["total_work"], plan["largest_tensor"], plan["largest_generated"])


def collect():
    entries = []
    contents = {}
    for case, (source, method, seed, queries) in CASES.items():
        files = ["order.json", "complexity.json"] + [f"query-{query}-order.json" for query in queries]
        scalar_plan = json.loads((ROOT / source / "order.json").read_text())
        for filename in files:
            source_path = ROOT / source / filename
            content = source_path.read_bytes()
            value = json.loads(content)
            if filename == "complexity.json":
                assert set(value) == {"steps", "total_work", "largest_tensor_entries",
                                      "largest_generated_entries", "units"}
                assert value["total_work"] == scalar_plan["total_work"]
                assert value["largest_tensor_entries"] == scalar_plan["largest_tensor"]
                assert value["largest_generated_entries"] == scalar_plan["largest_generated"]
                assert value["steps"] == scalar_plan["steps"]
            else:
                query = [] if filename == "order.json" else [int(filename.split("-")[1])]
                check_plan(value, method, seed, query)
            destination = f"{case}/{filename}"
            contents[destination] = content
            entries.append({"file": destination, "source": f"{source}/{filename}",
                            "sha256": hashlib.sha256(content).hexdigest()})
    return {"description": "Reviewed contraction plans and scalar complexity; no model values or raw run logs.",
            "files": entries}, contents


def verify_saved():
    manifest = json.loads((DESTINATION / "manifest.json").read_text())
    expected = {}
    for case, (source, method, seed, queries) in CASES.items():
        for filename in ["order.json", "complexity.json"] + [f"query-{v}-order.json" for v in queries]:
            expected[f"{case}/{filename}"] = f"{source}/{filename}"
        scalar = json.loads((DESTINATION / case / "order.json").read_text())
        for filename, query in [("order.json", [])] + [(f"query-{v}-order.json", [v]) for v in queries]:
            plan = json.loads((DESTINATION / case / filename).read_text())
            check_plan(plan, method, seed, query)
        cost = json.loads((DESTINATION / case / "complexity.json").read_text())
        assert cost["total_work"] == scalar["total_work"]
        assert cost["largest_tensor_entries"] == scalar["largest_tensor"]
        assert cost["largest_generated_entries"] == scalar["largest_generated"]
        assert cost["steps"] == scalar["steps"]
    entries = manifest["files"]
    assert len(entries) == len(expected)
    assert {entry["file"]: entry["source"] for entry in entries} == expected
    for entry in entries:
        data = (DESTINATION / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["file"]
    print(f"All {len(entries)} saved files pass independent cost and checksum verification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true",
                      help="replace the saved evidence with the current local run plans")
    mode.add_argument("--verify", action="store_true",
                      help="verify saved plans and checksums without local run data")
    args = parser.parse_args()
    if args.verify:
        verify_saved()
        return
    manifest, contents = collect()
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    if args.refresh:
        for name, content in contents.items():
            target = DESTINATION / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (DESTINATION / "manifest.json").write_bytes(manifest_bytes)
        print(f"Exported {len(contents)} verified plan and complexity files.")
    else:
        for name, content in contents.items():
            target = DESTINATION / name
            assert target.read_bytes() == content, f"Saved evidence differs from local run: {name}"
        assert (DESTINATION / "manifest.json").read_bytes() == manifest_bytes, "Manifest differs"
        print(f"All {len(contents)} saved plan and complexity files match the local runs.")


if __name__ == "__main__":
    main()
