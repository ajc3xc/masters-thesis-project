from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Root dataset directory
root = Path("D:/camerer_ml/datasets/Final-Dataset-Vol1")

# Original input files
split_files = {
    "train": root / "train.txt",
    "test": root / "test.txt",
}

# Process function
def process_split(name, path):
    out_lines = []

    # Read base filenames (e.g., "0001.png")
    with open(path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    # Generate new lines: "images/0001.png labels/0001.png"
    for fname in filenames:
        out_lines.append(f"images\\{fname}" + " " + f"labels\\{fname}")

    # Write to updated file
    updated_path = path.with_name(f"updated_{path.name}")
    with open(updated_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"✅ Wrote {updated_path.name} ({len(out_lines)} entries)")

# Run both in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_split, name, path) for name, path in split_files.items()]
    for f in futures:
        f.result()
