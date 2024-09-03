#!/usr/bin/env python3
import json
from enum import Enum

import numpy as np
import pandas as pd
import vtk

from ...enums import Plane, str2plane, plane2str
from ...metrics import lateral_error, euclidean_error
from ...paths import DATA_PATH


class Acquisition:
    def __init__(
        self, name: str, target: str, operator: str, plane: Plane, indices=None
    ):
        self.name = name
        self.target = target
        self.target_index = int(self.target[1:]) - 1
        self.operator = operator
        self.plane = plane
        self.indices = indices

    @staticmethod
    def from_string(s: str):
        tokens = s.strip().split(" ")
        indices = [int(t) for t in tokens[:-1] if t.strip()]
        descriptor = tokens[-1]
        trailing_minuses = 0
        while descriptor.startswith("-"):
            trailing_minuses += 1
            descriptor = descriptor[1:]
        indices.append(trailing_minuses)
        tokens = descriptor.strip().split("-")
        assert len(tokens) == 4
        target = tokens[0]
        _ = tokens[1]  # "cryo"
        operator = tokens[2]
        plane = tokens[3]
        # sometimes plane descriptor ends with a single number. just add it to index
        if plane[-1].isdigit():
            indices.append(int(plane[-1]))
            plane = plane[:-1]
        plane = str2plane(plane)
        return Acquisition(descriptor, target, operator, plane, indices)

    def row(self):
        return dict(
            name=self.name,
            target=self.target,
            Operator=self.operator,
            Plane=plane2str(self.plane),
            target_index=int(self.target[1:]),
        )

    def __str__(self):
        s = f"Acquisition {self.name}: target={self.target}, operator={self.operator}, plane={self.plane}"
        return s


def load_acquisitions():
    filename = DATA_PATH / "cryotrack_validation/acquisitions.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    return [Acquisition.from_string(line) for line in lines]


def load_tip_positions():
    with open(DATA_PATH / "cryotrack_validation/markups/tip.mrk.json", "r") as f:
        d = json.load(f)
    markups = d["markups"]
    controlPoints = markups[0]["controlPoints"]
    positions = {int(p["id"]): np.array(p["position"]) for p in controlPoints}
    return positions


def load_entry_points():
    with open(
        DATA_PATH / "cryotrack_validation/markups/entry-point.mrk.json", "r"
    ) as f:
        d = json.load(f)
    markups = d["markups"]
    controlPoints = markups[0]["controlPoints"]
    positions = {int(p["id"]): np.array(p["position"]) for p in controlPoints}
    return positions


def load_targets():
    with open(DATA_PATH / "cryotrack_validation/markups/target.mrk.json", "r") as f:
        d = json.load(f)
    markups = d["markups"]
    controlPoints = markups[0]["controlPoints"]
    positions = {int(p["id"]) - 1: np.array(p["position"]) for p in controlPoints}
    return positions


def load_tumor_meshes():
    model_path = DATA_PATH / "cryotrack_validation" / "models"
    tumor_paths = model_path.glob("tumor*.vtk")
    models = {}
    for tumor_path in tumor_paths:
        stem = tumor_path.stem
        target_index = int(stem[len("tumor-")]) - 1
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(tumor_path))
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        polydata = reader.GetOutput()
        models[target_index] = polydata
    return models


def load_risk_meshes():
    model_path = DATA_PATH / "cryotrack_validation/models"
    risk_structures = ["Airway", "Hepatic", "Portal"]
    models = {}
    for risk in risk_structures:
        risk_path = model_path / (risk.lower() + ".vtk")
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(risk_path))
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.Update()
        polydata = reader.GetOutput()
        models[risk] = polydata
    return models


def point_distance_to_polydata(point, polydata):
    implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
    implicitPolyDataDistance.SetInput(polydata)
    closestPoint = np.zeros(3)
    distance = implicitPolyDataDistance.EvaluateFunctionAndGetClosestPoint(
        point, closestPoint
    )
    return closestPoint, distance


def run_cryotrack_analysis() -> pd.DataFrame:
    acquisitions = load_acquisitions()
    target_points = load_targets()
    tip_positions = load_tip_positions()
    entry_points = load_entry_points()
    models = load_tumor_meshes()
    risk_models = load_risk_meshes()

    print("CRYOTRACK")

    rows = []
    for acquisition in acquisitions:
        row = acquisition.row()
        idx = acquisition.indices[0]
        tip_position = tip_positions[idx]
        entry_point = entry_points[idx]
        target_point = target_points[acquisition.target_index]
        model = models[acquisition.target_index]
        closestPoint, distance = point_distance_to_polydata(tip_position, model)

        for name, m in risk_models.items():
            _, tip_to_risk = point_distance_to_polydata(tip_position, m)
            row[f"D_{name}"] = tip_to_risk

        E_euclidean = euclidean_error(tip_position, target_point)
        # E_entry_euclidean = euclidean_error(entry_point, )
        E_lateral = lateral_error(target_point, entry_point, tip_position)
        E_tip_to_tumor = np.abs(distance)

        row.update(
            {
                "Euclidean Error (final)": E_euclidean,
                "Lateral Error (final)": E_lateral,
                "Euclidean (tip to tumor)": E_tip_to_tumor,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["D_risk_min"] = df[["D_" + name for name in risk_models.keys()]].min(1)
    return df
