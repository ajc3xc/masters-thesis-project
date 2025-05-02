from pathlib import Path
import os

# Print current working directory before
print("Before:", Path.cwd())

# Change working directory to the directory where this script is located
os.chdir(Path(__file__).resolve().parent)

# Print current working directory after
print("After:", Path.cwd())

# Try to open a file relative to the script
test_file = Path("test.py")
print("Looking for file:", test_file)

if test_file.exists():
    print("Found file! Contents:")
    print(test_file.read_text())
else:
    print("example.txt not found in script directory.")
