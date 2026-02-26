import json
from pathlib import Path

BASE_DIR = Path("4_ConstantAmplitude")
TARGET_FILENAME = "ProjectParameters_Custom.json"

CORRECT_LINE_NAMES = [
    "line1",
    "line2",
    "line3",
    "line4",
    "line5",
    "line6",
]

def fix_line_outputs(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    modified = False

    try:
        ascii_outputs = data["output_processes"]["ascii_output"]
    except KeyError:
        return False

    line_outputs = [
        p for p in ascii_outputs
        if p.get("process_name") == "LineOutputProcess"
    ]

    for idx, process in enumerate(line_outputs):
        if idx >= len(CORRECT_LINE_NAMES):
            break

        params = process.get("Parameters", {})
        file_settings = params.get("output_file_settings", {})

        expected_name = CORRECT_LINE_NAMES[idx]
        current_name = file_settings.get("file_name")

        if current_name != expected_name:
            file_settings["file_name"] = expected_name
            modified = True

    if modified:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    return modified


def main():
    modified_files = []
    
    for json_file in BASE_DIR.rglob(TARGET_FILENAME):
        if fix_line_outputs(json_file):
            modified_files.append(json_file)

    print("\n========== SUMMARY ==========")
    print(f"Files modified: {len(modified_files)}\n")

    for f in modified_files:
        print(f"✔ {f}")


if __name__ == "__main__":
    main()
