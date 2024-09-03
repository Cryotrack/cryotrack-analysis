#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context="paper", style="whitegrid", font_scale=1.2, rc={"text.usetex": True}
)

from cryotrack_analysis.insertion_analysis.cryotrack_validation import run_cryotrack_analysis
from cryotrack_analysis.insertion_analysis.CT_baseline import run_ctbaseline_analysis
from cryotrack_analysis.video_annotation.extract_bookmarks import extract_bookmarks_from_playlist
from cryotrack_analysis.video_annotation.extract_from_mha import read_timestamps_file


plot_path = Path("plots")
plot_path.mkdir(exist_ok=True)

tables_path = Path("tables")
tables_path.mkdir(exist_ok=True)

spreadsheets_path = Path("spreadsheets")
spreadsheets_path.mkdir(exist_ok=True)


latex_textwidth_LNCS = 347.12354  # in pt


def make_plots_accuracy(df_cryotrack, df_ctbaseline):
    figsize = (4.5, 2.6)

    df = df_cryotrack
    df = df[df["Operator"] != "JN"]  # exclude; only performed 1 or 2 insertions
    df = df.replace("JV", "S")
    df = df.replace("JM", "N1")
    df = df.replace("HK", "N2")
    # weird, but does the trick. I wanted the order to be "S, N1, N2"
    df = df.sort_values(by="Operator").sort_values(
        by="Operator", key=lambda x: x.str.len()
    )

    my_palette = {
        "S": "#81e9d8",
        "N1": "#b6b0ee",
        "N2": "#fff1a1"
    }
    my_palette = "Set3"

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean Error (final)",
        hue="Operator",
        palette=my_palette,
    )
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{With Cryotrack}")
    plt.ylim((0, 50))
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_euclidean_per_target.png", bbox_inches="tight", dpi=600
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="D_risk_min",
        hue="Operator",
        palette=my_palette,
    )
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Distance to Risk [mm]}")
    #plt.title("With Cryotrack")
    plt.ylim((0, 75))
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_riskdistance_per_target.png",
        bbox_inches="tight",
        dpi=600,
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Lateral Error (final)",
        hue="Operator",
        palette=my_palette,
    )
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Lateral Error [mm]}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.xticks([])
    plt.savefig(
        plot_path / "cryotrack_lateral_per_target.png", bbox_inches="tight", dpi=600
    )
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean (tip to tumor)",
        hue="Operator",
        palette=my_palette,
    )
    plt.xlabel("")#"Target ID")
    plt.ylabel(r"\textbf{Distance to Tumor [mm]}")
    plt.title(r"\textbf{With Cryotrack}")
    plt.ylim((0, 40))
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_tumor_per_target.png", bbox_inches="tight", dpi=600
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean Error (final)",
        hue="Plane",
        palette="Set3",
    )
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{With Cryotrack}")
    plt.ylim((0, 50))
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_euclidean_per_target_by_plane.png",
        bbox_inches="tight",
        dpi=600,
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Lateral Error (final)",
        hue="Plane",
        palette="Set3",
    )
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Lateral Error [mm]}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_lateral_per_target_by_plane.png",
        bbox_inches="tight",
        dpi=600,
    )

    #### CT baseline ####
    df = df_ctbaseline
    df = df.replace("JV", "S")
    figsize = (3.0, 2.6)

    ### Per target, grouped by operator
    ## Euclidean
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean Error (final)",
        hue="Operator",
        palette="Set3",
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{Without Cryotrack}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_euclidean_per_target.png", bbox_inches="tight", dpi=600
    )
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="D_risk_min",
        hue="Operator",
        palette="Set3",
    )
    plt.xlabel("Target ID")
    plt.ylabel("")#r"\textbf{Distance to closest risk [mm]}")
    #plt.title("Without Cryotrack")
    plt.ylim((0, 75))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_riskdistance_per_target.png",
        bbox_inches="tight",
        dpi=600,
    )
    ## Lateral
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df, x="target_index", y="Lateral Error", hue="Operator", palette="Set3"
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Lateral Error [mm]}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_lateral_per_target.png", bbox_inches="tight", dpi=600
    )
    ## Tumor distance
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean (tip to tumor)",
        hue="Operator",
        palette="Set3",
    )
    plt.xlabel("")#Target ID")
    plt.ylabel("")  # r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{Without Cryotrack}")
    plt.ylim((0, 40))
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_tumor_per_target.png", bbox_inches="tight", dpi=600
    )

    ### Per target, grouped by plane
    ## Euclidean
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean Error (final)",
        hue="Plane",
        palette="Set1",
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{Without Cryotrack}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_euclidean_per_target_by_plane.png",
        bbox_inches="tight",
        dpi=600,
    )
    ## Lateral
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df, x="target_index", y="Lateral Error", hue="Plane", palette="Set1"
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Lateral Error [mm]}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_lateral_per_target_by_plane.png",
        bbox_inches="tight",
        dpi=600,
    )

    ### Per target, grouped by strokes (ss / sw)
    ## Euclidean
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df,
        x="target_index",
        y="Euclidean Error (final)",
        hue="Strokes",
        palette="Set1",
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Euclidean Error [mm]}")
    plt.title(r"\textbf{Without Cryotrack}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_euclidean_per_target_by_strokes.png",
        bbox_inches="tight",
        dpi=600,
    )
    ## Lateral
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df, x="target_index", y="Lateral Error", hue="Strokes", palette="Set1"
    )
    plt.xlabel("Target ID")
    plt.ylabel("")  # r"\textbf{Lateral Error [mm]}")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_lateral_per_target_by_strokes.png",
        bbox_inches="tight",
        dpi=600,
    )


def make_plots_time(df_cryotrack_time, df_ctbaseline_time):
    figsize = (3.5, 2.7)
    df = df_cryotrack_time
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x="target_index", y="planning time [s]", palette="Blues")
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Planning time [s]}")
    sns.despine()
    plt.ylim((0, 750))
    plt.tight_layout()
    plt.savefig(plot_path / "cryotrack_planning_time_per_target.png", dpi=600)

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x="target_index", y="insertion time [s]", palette="Blues")
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Insertion time [s]}")
    sns.despine()
    plt.ylim((0, 750))
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_insertion_time_per_target.png",
        bbox_inches="tight",
        dpi=600,
    )
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x="target_index", y="total time [s]", palette="Blues")
    plt.xlabel("Target ID")
    plt.ylabel(r"\textbf{Total time [s]}")
    plt.ylim((0, 750))
    plt.title(r"\textbf{With Cryotrack}")
    plt.tight_layout()
    plt.savefig(
        plot_path / "cryotrack_duration_per_target.png", bbox_inches="tight", dpi=600
    )

    ##### CT BASELINE #####
    df = df_ctbaseline_time
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x="target_index", y="duration", palette="Blues")
    plt.xlabel("Target ID")
    plt.ylabel("")
    plt.title(r"\textbf{Without Cryotrack}")
    plt.ylim((0, 750))
    plt.tight_layout()
    plt.savefig(
        plot_path / "ctbaseline_duration_per_target.png", bbox_inches="tight", dpi=600
    )


def make_plots(df_cryotrack_time, df_ctbaseline_time, df_ctbaseline, df_cryotrack):
    make_plots_time(df_cryotrack_time, df_ctbaseline_time)
    make_plots_accuracy(df_cryotrack, df_ctbaseline)


def export_tables(df_cryotrack_time, df_ctbaseline_time, df_ctbaseline, df_cryotrack):
    d = {
        "Operator": [],
        "Plane": [],
        "Strokes": [],
        "Tumor distance [mm]": [],
        "Risk distance [mm]": [],
        "Total time [s]": []
    }

    for operator in ("S", "N1", "N2"):
        for plane in ("ip", "oop"):
            df_sub = df_cryotrack_time.replace("JV", "S")
            df_sub = df_sub.replace("JM", "N1")
            df_sub = df_sub.replace("HK", "N2")
            df_sub = df_sub[df_sub["Operator"] == operator]
            df_sub = df_sub[df_sub["Plane"] == plane]
            t_total = df_sub["total time [s]"].mean()

            p = plane if plane != "oop" else "op"
            df_sub = df_cryotrack.replace("JV", "S")
            df_sub = df_sub.replace("JM", "N1")
            df_sub = df_sub.replace("HK", "N2")
            df_sub = df_sub[df_sub["Operator"] == operator]
            df_sub = df_sub[df_sub["Plane"] == p]
            tip_to_tumor = df_sub["Euclidean (tip to tumor)"].mean()
            d_risk = df_sub["D_risk_min"].mean()

            d["Operator"].append(operator)
            d["Plane"].append(plane.upper())
            d["Strokes"].append(1)
            d["Tumor distance [mm]"].append(tip_to_tumor)
            d["Risk distance [mm]"].append(d_risk)
            d["Total time [s]"].append(t_total)
    
    df = pd.DataFrame(d)
    df = df.sort_values(by="Operator")
    print(df[["Tumor distance [mm]", "Risk distance [mm]", "Total time [s]"]].mean())

    styler = df.style.format(precision=2).hide(axis="index")
    styler.to_latex(tables_path / "cryotrack.tex")

    d = {
        "Operator": [],
        "Plane": [],
        "Strokes": [],
        "Tumor distance [mm]": [],
        "Risk distance [mm]": [],
        "Total time [s]": []
    }

    for Strokes in set(df_ctbaseline.Strokes):
        for plane in ("IP", "OoP"):
            df_sub = df_ctbaseline_time
            df_sub = df_sub[df_sub["Plane"] == plane]
            df_sub = df_sub[df_sub["Strokes"] == Strokes]
            t_total = df_sub["duration"].mean()

            p = plane.lower() if plane != "OoP" else "op"
            df_sub = df_ctbaseline
            df_sub = df_sub[df_sub["Plane"] == p]
            df_sub = df_sub[df_sub["Strokes"] == Strokes]
            tip_to_tumor = df_sub["Euclidean (tip to tumor)"].mean()
            d_risk = df_sub["D_risk_min"].mean()

            d["Operator"].append("JV")
            d["Plane"].append(plane.upper())
            d["Strokes"].append(1 if Strokes == "ss" else 3)
            d["Tumor distance [mm]"].append(tip_to_tumor)
            d["Risk distance [mm]"].append(d_risk)
            d["Total time [s]"].append(t_total)
    
    df = pd.DataFrame(d)
    df = df.sort_values(by="Strokes")
    print(df[["Tumor distance [mm]", "Risk distance [mm]", "Total time [s]"]].std())
    styler = df.style.format(precision=2).hide(axis="index")
    styler.to_latex(tables_path / "ctbaseline.tex")



def run_all_analyses():
    video_dfs = []
    for path in Path("data/cryotrack_validation/video_bookmarks").glob("*.xspf"):
        dfs = extract_bookmarks_from_playlist(path, exclude_invalid=True)
        video_dfs.extend(dfs)
    # These are the three dataframes to analyze:
    df_cryotrack_time = pd.concat(video_dfs)
    df_ctbaseline_time = read_timestamps_file("timestamps.json")
    df_ctbaseline = run_ctbaseline_analysis()
    df_cryotrack = run_cryotrack_analysis()

    # Export spreadsheets
    df_cryotrack_time.to_excel(spreadsheets_path / "cryotrack_time.xlsx")
    df_ctbaseline_time.to_excel(spreadsheets_path / "ctbaseline_time.xlsx")
    df_ctbaseline.to_excel(spreadsheets_path / "ctbaseline.xlsx")
    df_cryotrack.to_excel(spreadsheets_path / "cryotrack.xlsx")

    export_tables(df_cryotrack_time, df_ctbaseline_time, df_ctbaseline, df_cryotrack)
    make_plots(df_cryotrack_time, df_ctbaseline_time, df_ctbaseline, df_cryotrack)


if __name__ == "__main__":
    run_all_analyses()
