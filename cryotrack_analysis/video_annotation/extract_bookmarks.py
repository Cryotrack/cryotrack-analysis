#!/usr/bin/env python3
from typing import List, Dict
import xml.etree.ElementTree as ET

import click
import pandas as pd


def convert_time_string_to_ms(s: str) -> float:
    """
    Bookmark records in VLC playlists contain timestamps in the format
    seconds.milliseconds
    or
    seconds,milliseconds
    depending on system locale. This function converts this representation
    to a single float representing milliseconds.
    """
    # We might receive a trailing } due to the simplistic tokenization
    if s[-1] == "}":
        s = s[:-1]
    if "," in s:
        tokens = s.split(",")
    elif "." in s:
        tokens = s.split(".")
    seconds = int(tokens[0])
    ms = int(tokens[1])
    return seconds * 1000 + ms


def record_to_dict(record: List) -> Dict:
    name = record[0]
    t = record[1]
    tokens = name.split("_")
    phase = tokens[0]
    target = tokens[1]
    operator = tokens[2]
    plane = tokens[3]
    attempt = 1
    if len(tokens) > 4:
        attempt = tokens[4]
    return {
        "name": name,
        "t": t,
        "phase": phase,
        "target": target,
        "Operator": operator,
        "Plane": plane,
        "attempt": attempt,
    }


def parse_bookmarks_record(bookmarks: str) -> List[Dict]:
    records = bookmarks.split("},")
    records = [record[1:] for record in records]
    pairs = [record.split(",time=") for record in records]
    dicts = [
        record_to_dict((pair[0][len("name=") :], convert_time_string_to_ms(pair[1])))
        for pair in pairs
    ]
    return dicts


def group_insertions(dicts: List[Dict]) -> pd.DataFrame:
    """
    This code works under the assumption that we have sequences of planning, insertion start and
    insertion end. It is a finite state machine which sets t_P, t_S, t_E and at every "E"
    step saves these three values to a dict named "row". All rows are merged to a pandas
    DataFrame.
    """
    result = []
    t = {}
    for d in dicts:
        t[d["phase"]] = d["t"]
        # TODO use enum for P, S, E
        if d["phase"] == "E":
            planning_time = t["S"] - t["P"]
            insertion_time = t["E"] - t["S"]
            total_time = t["E"] - t["P"]
            row = {
                **d,
                "target_index": int(d["target"][1:]),
                "planning time [s]": planning_time / 1000,
                "insertion time [s]": insertion_time / 1000,
                "total time [s]": total_time / 1000,
            }
            del row["t"]
            del row["phase"]
            row["name"] = row["name"][2:]
            result.append(row)
    return result


def extract_bookmarks_from_playlist(filename: str, exclude_invalid=True) -> pd.DataFrame:
    """
    :param filename: Path to *.xspf file
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    playlist = root

    trackList = playlist.find("{http://xspf.org/ns/0/}trackList")
    tracks = trackList.findall("{http://xspf.org/ns/0/}track")
    # we typically only have a single track inside a playlist file
    dfs = []
    for track in tracks:
        extension = track.find("{http://xspf.org/ns/0/}extension")
        option = extension.find("{http://www.videolan.org/vlc/playlist/ns/0/}option")
        bookmarks = option.text[len("bookmarks=") :]
        dicts = parse_bookmarks_record(bookmarks)
        insertions = group_insertions(dicts)
        df = pd.DataFrame(insertions)
        if exclude_invalid:
            df = df[~df.name.str.contains("invalid")]
        dfs.append(df)
    return dfs