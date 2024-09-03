#!/usr/bin/env python3
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import vtk

from ...metrics import lateral_error, euclidean_error

# Load ground truth target positions. We will need them in PlannedTarget.load
# to determine if final and entry points have been mapped correctly.
with open("data/CT_baseline/markups/tumor.mrk.json", "r") as f:
    d = json.load(f)
    markups = d["markups"]
    controlPoints = markups[0]["controlPoints"]
    positions = [np.array(p["position"]) for p in controlPoints]
    tumor_points = np.array(positions)


class LineMarkup:
    def __init__(self, path, target_index):
        self.path = path
        self.target_index = target_index
        self.entry_point = None
        self.final_point = None
        self.load()

    def needle_vector(self):
        return self.final_point - self.entry_point

    def depth(self):
        return np.linalg.norm(self.needle_vector())

    def load(self):
        with open(self.path, "r") as f:
            d = json.load(f)
        markups = d["markups"]
        controlPoints = markups[0]["controlPoints"]
        # Sometimes entry point and final point are swapped.
        # A simple and safe heuristic is to check the distance between final point and actual target position.
        self.final_point = np.array(controlPoints[0]["position"])
        self.entry_point = np.array(controlPoints[1]["position"])
        if np.linalg.norm(
            tumor_points[self.target_index] - self.entry_point
        ) < np.linalg.norm(tumor_points[self.target_index] - self.final_point):
            self.final_point = np.array(controlPoints[1]["position"])
            self.entry_point = np.array(controlPoints[0]["position"])


class PlannedTarget(LineMarkup):
    def __init__(self, name, path, plane):
        self.name = name
        self.index = int(self.name[1:]) - 1
        self.plane = plane.lower()
        super(PlannedTarget, self).__init__(path, self.index)

    @staticmethod
    def is_target_markup_path(path):
        stem = path.stem[: -len(".mrk")]
        return re.match("^t[0-9]-(IP|OoP|OP|OOP)$", stem) is not None

    @staticmethod
    def from_path(path):
        stem = path.stem[: -len(".mrk")]
        tokens = stem.split("-")
        name = tokens[0]
        plane = tokens[1].lower()
        if plane == "oop":
            plane = "op"
        return PlannedTarget(name, path, plane)

    def __str__(self):
        return f"PlannedTarget {self.name}: plane={self.plane}"


class Insertion(LineMarkup):
    def __init__(self, index, path, target, plane, strokes, attempt=0):
        self.index = index
        self.path = path
        self.target = target
        self.index = int(self.target[1:]) - 1
        self.plane = plane
        self.strokes = strokes
        self.attempt = attempt
        super(Insertion, self).__init__(path, self.index)

    def row(self):
        return dict(
            name=self.path.stem[: -len(".mrk")],
            Plane=self.plane,
            target=self.target,
            Strokes=self.strokes,
            target_index=int(self.target[1:])
        )

    @staticmethod
    def is_insertion_markup_path(path):
        stem = path.stem[: -len(".mrk")]
        return (
            re.match("^[0-9]?[0-9] T[0-9]-(IP|OoP|OP|OOP)-(sw|ss)-[0-9]$", stem)
            is not None
        )

    @staticmethod
    def from_path(path):
        stem = path.stem[: -len(".mrk")]
        tokens = stem.replace("-", " ").split(" ")
        index = int(tokens[0])
        target = tokens[1].lower()
        plane = tokens[2].lower()
        strokes = tokens[3].lower()
        attempt = int(tokens[4])
        return Insertion(index, path, target, plane, strokes, attempt)

    def __str__(self):
        return f"Insertion {self.index}: target={self.target} plane={self.plane} strokes={self.strokes} attempt={self.attempt}"


def load_tumor_meshes():
    model_path = Path("data") / "CT_baseline" / "models"
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
    model_path = Path("data") / "CT_baseline" / "models"
    risk_structures = ["Airway", "Hepatic", "Portal"]#, "Liver", "Lungs"]
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


def run_ctbaseline_analysis() -> pd.DataFrame:
    insertions = []
    targets = {}

    for p in Path("data/CT_baseline/markups/").glob("*.mrk.json"):
        if Insertion.is_insertion_markup_path(p):
            insertions.append(Insertion.from_path(p))
        if PlannedTarget.is_target_markup_path(p):
            t = PlannedTarget.from_path(p)
            targets[(t.name, t.plane)] = t

    print("CT BASELINE")

    tumor_models = load_tumor_meshes()
    risk_models = load_risk_meshes()

    # convert to pandas dataframe and save it
    rows = []
    for insertion in insertions:
        row = insertion.row()
        target_index = (insertion.target, insertion.plane)
        target = targets[target_index]

        tumor_model = tumor_models[insertion.index]
        closestPoint, distance = point_distance_to_polydata(insertion.final_point, tumor_model)
        
        E_tip_to_tumor = np.abs(distance)
        E_final_euclidean = euclidean_error(target.final_point, insertion.final_point)
        E_entry_euclidean = euclidean_error(target.entry_point, insertion.entry_point)
        E_lateral = lateral_error(
            target.final_point, insertion.entry_point, insertion.final_point
        )
        for name, m in risk_models.items():
            _, tip_to_risk = point_distance_to_polydata(insertion.final_point, m)
            row[f"D_{name}"] = tip_to_risk

        row.update(
            {
                "Operator": "JV",
                "Euclidean Error (final)": E_final_euclidean,
                "Entry Point Error": E_entry_euclidean,
                "Euclidean (tip to tumor)": E_tip_to_tumor,
                "Lateral Error": E_lateral,
                "Target Depth": target.depth()
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["D_risk_min"] = df[["D_" + name for name in risk_models.keys()]].min(1)
    return df
