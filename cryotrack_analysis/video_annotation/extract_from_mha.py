from pathlib import Path
import json
import pandas as pd


def extract_timestamps_from_sequences(input_folder):
    """
    Aggregate mha files and save start/end time stamps to a dict.
    """
    file_names = []
    start_timestamps = []
    end_timestamps = []

    # Timestamps are written like : Seq_Frame*_Timestamp = value
    # For each file, get the name, starting time and end time
    for filename in Path(input_folder).glob("*.mha"):
        file_names.append(filename.stem)
        with open(filename, "r") as f:
            lines = f.readlines()
            # find the first instance of Seq_Frame*_Timestamp
            start_timestamp = None
            end_timestamp = None
            for line in lines:
                if "Seq_Frame" in line and "Timestamp" in line:
                    start_timestamp = float(line.split("=")[1].strip())
                    break

            # find the last instance of Seq_Frame*_Timestamp
            for line in reversed(lines):
                if "Seq_Frame" in line and "Timestamp" in line:
                    end_timestamp = float(line.split("=")[1].strip())
                    break

            start_timestamps.append(start_timestamp)
            end_timestamps.append(end_timestamp)

    # Create a dict with the data
    # (file_name, start_timestamp, end_timestamp, duration)
    data = {}

    for i in range(len(file_names)):
        data[file_names[i]] = {
            "start_timestamp": start_timestamps[i],
            "end_timestamp": end_timestamps[i],
            "duration": end_timestamps[i] - start_timestamps[i],
        }

    return data


def read_timestamps_file(filename, data_path="data/CT_baseline") -> pd.DataFrame:
    path = Path(data_path) / filename
    with open(path, "r") as f:
        d = json.load(f)
    rows = []
    for key, value in d.items():
        descriptor, _ = key.split(".")
        tokens = descriptor.split("-")
        target = tokens[0]
        target_index = int(target[1:])
        plane = tokens[1]
        strokes = "ss"
        if len(tokens) > 2 and tokens[2] in ("ss", "sw"):
            strokes = tokens[2]
        row = dict(
            name=descriptor,
            target=target,
            target_index=target_index,
            Plane=plane,
            Strokes=strokes,
            operator="JV",
            start_timestamp=value["start_timestamp"],
            end_timestamp=value["end_timestamp"],
            duration=value["duration"]
        )
        rows.append(row)
    return pd.DataFrame(rows)