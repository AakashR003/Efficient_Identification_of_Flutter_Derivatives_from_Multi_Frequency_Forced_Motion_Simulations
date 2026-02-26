import json
from pathlib import Path
import shutil

# =========================
# USER SETTINGS
# =========================
BASE_DIR = Path("6_ConstantAmplitude")
TARGET_FILENAME = "ProjectParameters_Custom.json"

OLD_AMPLITUDE = 0.1047
NEW_AMPLITUDE = 0.14

# =========================
# SCRIPT
# =========================
def update_amplitude(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    modified = False

    try:
        aux_list = data["processes"]["auxiliar_process_list"]
    except KeyError:
        return False  # structure not present

    for process in aux_list:
        if process.get("process_name") == "ForcedMotionWithDistortionProcess":
            imposed_motion = process.get("Parameters", {}).get("imposed_motion", {})
            amplitudes = imposed_motion.get("amplitudes", [])

            if isinstance(amplitudes, list) and OLD_AMPLITUDE in amplitudes:
                imposed_motion["amplitudes"] = [
                    NEW_AMPLITUDE if a == OLD_AMPLITUDE else a
                    for a in amplitudes
                ]
                modified = True

    if modified:
        # Backup original file
        #backup_path = json_path.with_suffix(".json.bak")
        #shutil.copy2(json_path, backup_path)

        # Write updated JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    return modified


def main():
    json_files = BASE_DIR.rglob(TARGET_FILENAME)

    modified_files = []

    for json_file in json_files:
        if update_amplitude(json_file):
            modified_files.append(json_file)

    print("\n========== SUMMARY ==========")
    print(f"Total files modified: {len(modified_files)}\n")

    for f in modified_files:
        print(f"✔ {f}")

    print("\nBackups saved as *.json.bak")


if __name__ == "__main__":
    main()
