#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION 7

1) Reads per‐subject Excel landmarks for 3D scans & moulds to compute ALP deviations,
   centroid trajectories, surface deviation maps, and iliac width/tilt angles (Figures 1–3).
2) Runs the combined brace‐design pipeline with updated PCA (auto component selection),
   leave‐one‐out testing, synthetic shape generation, mean‐shape error analysis, and CSA,
   following Dickinson et al., Prosthesis 2021;3:280–299.
3) (NEW) Generates 2D coronal cross‐section images at the iliac crest for each patient's post‐rectified mesh,
   extracts HOG features, reduces via PCA, runs K‐means clustering, and compares clusters to:
     • Binary Cobb‐improvement label (>=10° thoracic improvement)
     • Multiclass “Dominant Correction Region” (Thoracic / Lumbar / Thoraco‐Lumbar)
"""

import os
import sys
import time
import csv
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from skimage.feature import hog

import open3d as o3d
import vtk
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ampscan imports (make sure ampscan is installed and on PYTHONPATH)
from ampscan import AmpObject, align, analyse
from ampscan.registration import registration
from ampscan.analyse import CMapOut
from ampscan.vis import qtVtkWindow

# PyQt for the VTK viewer (used only if interactive)
from PyQt5.QtWidgets import QApplication

# ──────────────────────────────────────────────────────────────────────────────
#                      GLOBAL CONFIGURATION & UTILS
# ──────────────────────────────────────────────────────────────────────────────

# Custom global color palette for consistent plotting
REGION_PALETTE = {
    "Torso": "tab:red",
    "Pelvis": "tab:blue",
    "CLAV": "tab:red",
    "STRN": "tab:green",
    "L-ASIS": "tab:blue",
    "MID": "tab:orange"
}

# Additional palettes for PCA/SSM plots
PCA_PALETTE = {
    "pre": "tab:red",
    "post": "tab:blue",
    "combined": "tab:green"
}

# Global Seaborn theme
sns.set_theme(style="whitegrid", rc={
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
})

# Adjust these paths to match your directory structure:
PRE_STL_DIR     = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\Landmark_3D_Scans_STL"
POST_STL_DIR    = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\Landmark_Moulds_STL"
EXCEL_SCAN_DIR  = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\Excel_Landmarks\Excel_Scans"
EXCEL_MOULD_DIR = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\Excel_Landmarks\Excel_Moulds"

# Number of subjects (adjust if you have more/less)
N_SUBJECTS = 11

# Output directory for all figures & CSVs
OUTPUT_DIR = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\figures9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map Excel “Landmark” strings → standardized keys
AOP_TO_KEY = {
    "Xiphoid":                      "CLAV",
    "TL Apex":                      "ApexCurveR",
    "Right Greater Trochanter":     "GT-R",
    "Left Greater Trochanter":      "GT-L",
    "Left Anterior Superior Iliac Spine":  "L-ASIS",
    "Right Anterior Superior Iliac Spine": "R-ASIS",
    "Left Under Arm":               "UnderarmL",
    "Right Under Arm":              "UnderarmR",
    "Left Posterior Superior Iliac Spine": "L-PSIS",
    "Right Posterior Superior Iliac Spine": "R-PSIS",
    "Symphysis Pubis":              "PUBIS",
    "Sternal Notch":                "STRN",
    "T3":                           "T3",
    "T8":                           "T8",
    "C7":                           "C7",
    "L4":                           "L4",
    "Left Coracoid process":        "CoracoidL",
    "Right Coracoid process":       "CoracoidR",
    "Left Scapular Spine":          "ScapSpineL",
    "Right Scapular Spine":         "ScapSpineR",
    "Umbilicus":                    "UMBILICUS"
}

# ──────────────────────────────────────────────────────────────────────────────
#                    SECTION 0: EXCEL METADATA READ & PROCESS
# ──────────────────────────────────────────────────────────────────────────────

def read_landmarks_from_excel(fn):
    """
    Read a single subject’s Excel file of 3D‐landmarks.
    Returns: dict mapping standardized key → (X, Y, Z)
    """
    df = pd.read_excel(fn, engine="openpyxl")
    lm = {}
    for _, row in df.iterrows():
        raw = str(row["Landmark"]).strip()
        if raw in AOP_TO_KEY:
            lm[AOP_TO_KEY[raw]] = (row["X"], row["Y"], row["Z"])
    return lm

def build_subject_landmark_dicts():
    """
    Builds two dicts: pre_lm[i] and post_lm[i] for i in 1..N_SUBJECTS,
    where each is a dict of standardized landmarks.
    Only includes subjects with both pre‐ and post‐ Excel files and valid ASIS pair.
    """
    pre_lm, post_lm, subj_keys = {}, {}, {}
    for i in range(1, N_SUBJECTS + 1):
        scan_fn  = os.path.join(EXCEL_SCAN_DIR,  f"Scan_{i}.xlsx")
        mould_fn = os.path.join(EXCEL_MOULD_DIR, f"Mould_{i}.xlsx")
        if not os.path.exists(scan_fn) or not os.path.exists(mould_fn):
            print(f"Subject {i}: missing Excel file, skipping.")
            continue
        p = read_landmarks_from_excel(scan_fn)
        m = read_landmarks_from_excel(mould_fn)
        common = set(p.keys()) & set(m.keys())
        if {"R-ASIS", "L-ASIS"} <= common:
            pre_lm[i], post_lm[i], subj_keys[i] = p, m, common
        else:
            print(f"Subject {i}: missing ASIS pair, skipping.")
    valid = sorted(subj_keys.keys())
    return pre_lm, post_lm, valid

# Build the landmark dicts once (used in ALP + clustering)
pre_lm, post_lm, VALID_SUBJS = build_subject_landmark_dicts()
if not VALID_SUBJS:
    print("No subjects qualified for ALP analysis.")

# ──────────────────────────────────────────────────────────────────────────────
#                    SECTION 1: ALP DEVIATION & PLOTS (FIGURES 1–3)
# ──────────────────────────────────────────────────────────────────────────────

def align_to_pelvis(obj, lm):
    """
    Translates & rotates the mesh so that the midpoint of R‐ASIS & L‐ASIS is at the origin,
    and aligns that line with the X‐axis.
    """
    R = np.array(lm["R-ASIS"])
    L = np.array(lm["L-ASIS"])
    mid = 0.5 * (R + L)
    obj.vert -= mid
    vec = L - R
    θ = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-θ), np.sin(-θ)
    rot = np.array([[ c, -s, 0],
                    [ s,  c, 0],
                    [ 0,  0, 1]])
    obj.vert = obj.vert.dot(rot.T)
    return obj

def section_z(lm, sec):
    """
    Given a landmark dictionary (lm) and a key (like "L-ASIS" or "STRN" or "MID"),
    returns the Z‐coordinate. For "MID", returns midpoint Z between STRN & L‐ASIS.
    """
    if sec == "MID":
        if "STRN" in lm and "L-ASIS" in lm:
            return 0.5 * (lm["STRN"][2] + lm["L-ASIS"][2])
        return None
    return lm.get(sec, (None, None, None))[2]

def extract_contour(obj, z):
    """
    Returns the 3D point cloud of the cross‐section slice at z ± 1 mm
    using ampscan.analyse.create_slices.
    """
    sl = analyse.create_slices(obj, [z-1, z+1], 50, typ="real_intervals", axis=2)
    return np.vstack(sl[0]) if (len(sl) > 0 and len(sl[0]) > 0) else np.empty((0,3))

def run_alp_pipeline():
    """
    Generates Figures 1–3 (ALP deviations, centroids, max surface deviations).
    """
    # 1) Compute ALP deviations & boxplots (Figure 1)
    records = []
    for i in VALID_SUBJS:
        p = pre_lm[i]
        m = post_lm[i]
        common = sorted(set(p.keys()) & set(m.keys()))
        for didx, dim in enumerate(("dX","dY","dZ")):
            for lm_key in common:
                delta = m[lm_key][didx] - p[lm_key][didx]
                records.append({
                    "subject": i,
                    "landmark": lm_key,
                    "dim": dim,
                    "value": delta
                })
    alp_df = pd.DataFrame(records)
    
    dim_labels = {
        "dX": "ΔX\" - Medial/Lateral (mm) (a)",
        "dY": "ΔY\" - Anterior/Posterior (mm) (b)",
        "dZ": "ΔZ\" - Cranial/Caudal (mm) (c)"
    }
    plt.figure(figsize=(14, 6), facecolor='white')
    for j, dim in enumerate(("dX","dY","dZ")):
        ax = plt.subplot(1, 3, j + 1)
        ax.set_facecolor('white')
        sns.boxplot(
            x="landmark",
            y="value",
            data=alp_df.query("dim == @dim"),
            order=sorted(set(alp_df["landmark"])),
            ax=ax,
            showmeans=True,
            meanprops={
                "marker": "x",
                "markerfacecolor": "white",
                "markeredgecolor": "white",
                "markersize": 4,
                "linestyle": "none"
            }
        )
        ax.set_title("", pad=0)
        ax.set_xlabel("")
        ax.set_ylabel(dim_labels[dim])
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure1_ALP_deviations.png"), dpi=300, facecolor='white')
    plt.close()

    # 2) Centroid Locations & Trajectories (Figure 2)
    SECTIONS = ["L-ASIS", "MID", "STRN", "CLAV"]
    section_colors = {
        "L-ASIS": REGION_PALETTE["L-ASIS"],
        "MID":    REGION_PALETTE["MID"],
        "STRN":   REGION_PALETTE["STRN"],
        "CLAV":   REGION_PALETTE["CLAV"]
    }
    centroids = {sec: {"pre": [], "post": []} for sec in SECTIONS}
    for i in VALID_SUBJS:
        pre_obj  = AmpObject(os.path.join(PRE_STL_DIR,  f"3D Scan {i}.stl"))
        post_obj = AmpObject(os.path.join(POST_STL_DIR, f"Mould {i}.stl"))
        pre_obj  = align_to_pelvis(pre_obj, pre_lm[i])
        post_obj = align_to_pelvis(post_obj, post_lm[i])
        for sec in SECTIONS:
            z = section_z(pre_lm[i], sec)
            if z is None:
                continue
            slp = analyse.create_slices(pre_obj,  [z-1, z+1], 50, typ="real_intervals", axis=2)
            slm = analyse.create_slices(post_obj, [z-1, z+1], 50, typ="real_intervals", axis=2)
            # explicit length checks instead of 'if slp and slp[0] and slm and slm[0]'
            if len(slp) > 0 and len(slp[0]) > 0 and len(slm) > 0 and len(slm[0]) > 0:
                centroids[sec]["pre"].append(np.vstack(slp[0]).mean(axis=0))
                centroids[sec]["post"].append(np.vstack(slm[0]).mean(axis=0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(top=0.88, wspace=0.4, hspace=0.4)

    sections_order = ["L-ASIS", "MID", "STRN", "CLAV"]
    for idx, sec in enumerate(sections_order):
        ax = axes[idx//3, idx%3]
        ax.set_facecolor('white')
        a = np.array(centroids[sec]["pre"])
        b = np.array(centroids[sec]["post"])
        if a.size > 0:
            ax.scatter(a[:, 0], a[:, 1],
                       label="Pre-rectification",
                       alpha=0.6,
                       color=section_colors[sec])
        if b.size > 0:
            ax.scatter(b[:, 0], b[:, 1],
                       marker="x",
                       label="Post-rectification",
                       color=section_colors[sec])
        ax.set_title(f"({chr(97 + idx)}) {sec} section")
        ax.set_xlabel("X coordinate (mm)")
        ax.set_ylabel("Y coordinate (mm)")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(True)

    # Panel (e): Posterior view (X vs Z)
    ax = axes[1,1]
    ax.set_facecolor('white')
    for sec in SECTIONS:
        a = np.array(centroids[sec]["pre"])
        b = np.array(centroids[sec]["post"])
        if a.size > 0 and b.size > 0:
            for p, q in zip(a, b):
                ax.arrow(p[0], p[2],
                         q[0] - p[0], q[2] - p[2],
                         head_width=3,
                         length_includes_head=True,
                         color=section_colors[sec],
                         alpha=0.8)
    ax.set_title("(e) Posterior view")
    ax.set_xlabel("X coordinate (mm)")
    ax.set_ylabel("Z coordinate (mm)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True)

    # Panel (f): Lateral view (Y vs Z)
    ax = axes[1,2]
    ax.set_facecolor('white')
    for sec in SECTIONS:
        a = np.array(centroids[sec]["pre"])
        b = np.array(centroids[sec]["post"])
        if a.size > 0 and b.size > 0:
            for p, q in zip(a, b):
                ax.arrow(p[1], p[2],
                         q[1] - p[1], q[2] - p[2],
                         head_width=3,
                         length_includes_head=True,
                         color=section_colors[sec],
                         alpha=0.8)
    ax.set_title("(f) Lateral view")
    ax.set_xlabel("Y coordinate (mm)")
    ax.set_ylabel("Z coordinate (mm)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True)

    # Build legend handles
    section_handles = [
        plt.Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor=section_colors[sec],
                   markersize=8,
                   label=sec)
        for sec in SECTIONS
    ]
    scan_handles = [
        plt.Line2D([0], [0], marker='o', color='k', linestyle='',
                   markersize=8, label='Pre-rectification'),
        plt.Line2D([0], [0], marker='x', color='k', linestyle='',
                   markersize=12, label='Post-rectification')
    ]
    all_handles = section_handles + scan_handles

    # Place a single legend above the grid
    fig.legend(
        handles=all_handles,
        title="Section & Scan Type",
        ncol= len(all_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
        title_fontsize=14,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_centroids.png"),
                dpi=300,
                facecolor='white')
    plt.close()


    # ─── Fig 3: Max Surface Deviations by Section ─────────────────────────────
    section_names = ["CLAV","STRN","L-ASIS","GT-R","GT-L"]
    sec_max = {sec: [] for sec in section_names}
    for i in VALID_SUBJS:
        pre_obj  = AmpObject(os.path.join(PRE_STL_DIR,  f"3D Scan {i}.stl"))
        post_obj = AmpObject(os.path.join(POST_STL_DIR, f"Mould {i}.stl"))
        align(pre_obj, post_obj).runICP()
        reg = registration(pre_obj, post_obj, steps=5, smooth=1).reg
        dev = np.linalg.norm(reg.vert - pre_obj.vert, axis=1)
        plt.close('all')
        for sec in section_names:
            z = section_z(pre_lm[i], sec)
            if z is None:
                continue
            sl = analyse.create_slices(reg, [z - 1, z + 1], 50, typ="real_intervals", axis=2)
            if len(sl) > 0 and len(sl[0]) > 0:
                idxs = np.hstack(sl[0]).astype(int)
                sec_max[sec].append(dev[idxs].max())

    df7 = pd.DataFrame({sec: pd.Series(vals) for sec, vals in sec_max.items()})
    plt.figure(figsize=(8, 4), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    sns.boxplot(data=df7, palette="pastel")
    plt.ylabel("Max deviation (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure3_max_dev.png"), dpi=300, facecolor='white')
    plt.close()

    # Update palette for Figure 3 using a list of colors matching section order
    fig3_colors = [
        REGION_PALETTE["CLAV"],
        REGION_PALETTE["STRN"],
        REGION_PALETTE["L-ASIS"],
        "tab:orange", "tab:brown"
    ]
    plt.figure(figsize=(8, 4), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    sns.boxplot(data=df7, palette=fig3_colors)
    plt.ylabel("Max deviation (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure3_max_dev.png"), dpi=300, facecolor='white')
    plt.close()

    print("Completed Figures 1–3 (ALP Deviations, Centroids, Max Deviations).")

# ─── Revised Figure 5: Volumetric Deviations by Side & Region (2×2 with a–d labels) ──────────────
TORSO_LMS  = ["C7","T3","T8","CoracoidL","CoracoidR","CLAV","STRN"]
PELVIS_LMS = ["L-ASIS","R-ASIS","L-PSIS","R-PSIS","GT-L","GT-R","UMBILICUS"]

records = []
for i in VALID_SUBJS:
    p = pre_lm[i]
    m = post_lm[i]
    common = sorted(set(p.keys()) & set(m.keys()))
    for didx, dim in enumerate(("dX","dY")):
        for lm in common:
            delta = m[lm][didx] - p[lm][didx]
            records.append({
                "subject": i,
                "landmark": lm,
                "region": "Pelvis" if lm in PELVIS_LMS else "Torso",
                "dim": dim,
                "value": delta
            })
dev_df = pd.DataFrame(records)
dev_df["ap_side"] = np.where(
    dev_df["dim"] == "dX", 
    np.nan,
    np.where(dev_df["value"] >= 0, "Anterior", "Posterior")
)

df_convex  = dev_df[(dev_df["dim"] == "dX") & (dev_df["value"] >= 0)].copy()
df_convex["panel"] = "(f) Convex side\nΔX – Medial/Lateral (mm)"

df_concave = dev_df[(dev_df["dim"] == "dX") & (dev_df["value"] <  0)].copy()
df_concave["panel"] = "(g) Concave side\nΔX – Medial/Lateral (mm)"

df_ant     = dev_df[(dev_df["dim"] == "dY") & (dev_df["ap_side"] == "Anterior")].copy()
df_ant["panel"] = "(h) Anterior side\nΔY – Anterior/Posterior (mm)"

df_post    = dev_df[(dev_df["dim"] == "dY") & (dev_df["ap_side"] == "Posterior")].copy()
df_post["panel"] = "(i) Posterior side\nΔY – Anterior/Posterior (mm)"

fig5_df = pd.concat([df_convex, df_concave, df_ant, df_post], ignore_index=True)

ylim_dy = (-35, 30)
ylim_dx = (-35, 30)

landmark_order = ["CLAV", "STRN", "L-ASIS", "R-ASIS", "GT-R"]
panels = [
    "(f) Convex side\nΔX – Medial/Lateral (mm)",
    "(g) Concave side\nΔX – Medial/Lateral (mm)",
    "(h) Anterior side\nΔY – Anterior/Posterior (mm)",
    "(i) Posterior side\nΔY – Anterior/Posterior (mm)"
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=False)
plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)

for idx, panel_label in enumerate(panels):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # 1) Add subplot letter “(a)”, “(b)”, etc.
    letter = f"({chr(97 + idx)})"  # 97 == 'a'
    ax.text(
        0.02, 0.97,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top'
    )

    subset = fig5_df[fig5_df["panel"] == panel_label]
    if subset.empty:
        ax.set_visible(False)
        continue

    # 2) Decide whether this is ΔX or ΔY, set y‐limits and ylabel accordingly
    if "ΔX" in panel_label:
        ax_ylim = ylim_dx
        ylabel = "Medial/Lateral (mm)"
    else:
        ax_ylim = ylim_dy
        ylabel = "Anterior/Posterior (mm)"

    sns.boxplot(
        x="landmark",
        y="value",
        hue="region",
        data=subset,
        order=landmark_order,
        hue_order=["Torso", "Pelvis"],
        palette=REGION_PALETTE,
        ax=ax,
        showfliers=False,
        whis=[5, 95],
    )
    sns.stripplot(
        x="landmark",
        y="value",
        hue="region",
        data=subset,
        order=landmark_order,
        hue_order=["Torso", "Pelvis"],
        palette=REGION_PALETTE,
        dodge=True,
        marker="o",
        edgecolor="black",
        facecolor="white",
        size=4,
        alpha=0.7,
        ax=ax,
        linewidth=0.5,
        legend=False,
    )

    ax.set_ylim(ax_ylim)
    short_title = panel_label.split(") ")[1].split("\n")[0]
    ax.set_title(short_title, fontsize=12, pad=6)
    ax.set_ylabel(ylabel)

    # 3) **Force an “Landmark” label on every subplot**
    ax.set_xlabel("Landmark")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

### 4) Build a single legend at the very top
##handles, labels = axes[0,0].get_legend_handles_labels()
##unique = dict(zip(labels, handles))
##fig.legend(
##    unique.values(),
##    unique.keys(),
##    loc="upper center",
##    title="Region",
##    bbox_to_anchor=(0.5, 0.98),
##    ncol=2
##)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "Figure5_f_g_h_i.png"), dpi=300)
plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#             SECTION 2: PCA / SSM / Synthetic Shapes / LOO / CSA (FIGURE 4 etc.)
# ──────────────────────────────────────────────────────────────────────────────

def build_pca_model_auto(dataset, var_fraction=0.95):
    """
    Runs PCA with n_components chosen to explain at least var_fraction of variance.
    Returns: (pca, transformed_data, explained_variance_ratio_, scaler)
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset)
    pca = PCA(n_components=var_fraction)
    data_pca = pca.fit_transform(data_scaled)
    print(f"> Auto-PCA: {pca.n_components_} comps, variance={pca.explained_variance_ratio_.sum():.2f}")
    return pca, data_pca, pca.explained_variance_ratio_, scaler

def plot_pca_3d_scatter(pca_data, ids, n_show=100, labels=None):
    """
    Creates a 3D scatter of the first three principal components using custom color.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = min(n_show, pca_data.shape[0])
    for i in range(n):
        ax.scatter(
            pca_data[i,0], pca_data[i,1], pca_data[i,2],
            c=PCA_PALETTE["combined"], s=20, alpha=0.8
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D Scatter of First Three PCs")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scatter_three_PCs_custom_color.png"), dpi=300)
    plt.close()

def leave_one_out_test(dataset, n_components=None):
    """
    Performs leave‐one‐out reconstruction error for PCA with n_components.
    Returns array of reconstruction errors for each “left‐out” sample.
    """
    errors = []
    n = dataset.shape[0]
    k = n_components or (n - 1)
    for i in range(n):
        train = np.delete(dataset, i, axis=0)
        test  = dataset[i]
        scaler = StandardScaler()
        train_s = scaler.fit_transform(train)
        test_s  = scaler.transform(test.reshape(1, -1))
        pca = PCA(n_components=k)
        pca.fit(train_s)
        coeffs = (test_s - pca.mean_) @ pca.components_.T
        recon  = pca.mean_ + coeffs @ pca.components_
        errors.append(np.linalg.norm(test_s - recon) / (np.linalg.norm(test_s) + 1e-8))
    return np.array(errors)

def loo_error_for_n_components(dataset, max_components=11):
    """
    Returns an array of mean LOO errors for each k from 1..max_c,
    where max_c = min(max_components, n_samples - 1).
    """
    n = dataset.shape[0]
    max_c = min(max_components, n - 1)
    errs = []
    for k in range(1, max_c + 1):
        e = leave_one_out_test(dataset, n_components=k)
        errs.append(e.mean())
    return np.array(errs)

def mirror_shape(obj):
    """
    Flips the mesh along the X‐axis (useful if needed for left/right symmetry).
    """
    obj.vert[:, 0] *= -1
    return obj

def rotate_to_coronal_plane_v11(vertices):
    """
    Estimates the principal direction among the most posterior points to align 
    the mesh’s coronal plane with the XY plane. Returns the rotated vertices array.
    """
    posterior = vertices[:,1] > 0.95 * vertices[:,1].max()
    pts = vertices[posterior][:, [0,1]]
    pca = PCA(n_components=2).fit(pts)
    angle = np.arctan2(pca.components_[0,1], pca.components_[0,0])
    R = np.array([
        [np.cos(-angle), -np.sin(-angle), 0],
        [np.sin(-angle),  np.cos(-angle), 0],
        [0, 0, 1]
    ])
    print(f"  Applied rotation {np.degrees(-angle):.1f}° to coronal plane")
    return vertices @ R.T

def hc_laplacian_smooth_v11(vertices, faces, iterations=5, lambda_factor=0.5, mu_factor=-0.53):
    """
    Humphrey‐Cooch Laplacian smoothing for nonrigid surface refinement.
    Returns the smoothed vertex array.
    """
    for _ in range(iterations):
        nbrs = {i: set() for i in range(len(vertices))}
        for f in faces:
            nbrs[f[0]].update(f[1:])
            nbrs[f[1]].update((f[0], f[2]))
            nbrs[f[2]].update(f[:2])
        lap = vertices.copy()
        for i, neigh in nbrs.items():
            lap[i] = (1 - lambda_factor) * vertices[i] + lambda_factor * vertices[list(neigh)].mean(axis=0)
        for i, neigh in nbrs.items():
            lap[i] += mu_factor * (vertices[i] - lap[list(neigh)].mean(axis=0))
        vertices = lap
    return vertices

def sample_mesh(obj, target_points=30000):
    """
    Uniformly samples “target_points” points from the AmpObject mesh
    and returns them as a sorted flat vector (for SSM input).
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(obj.vert)
    mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=target_points)
    pts = np.asarray(pcd.points)
    idx = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
    return pts[idx].flatten()

def adaptive_trimline(vertices, faces,
                      lower_trim_fraction=0.20,
                      upper_trim_fraction=0.85,
                      min_vertex_threshold=100):
    """
    Trims away vertices outside the [lower_trim_fraction, upper_trim_fraction] 
    along Z, discards faces that fall outside, and returns the trimmed mesh.
    """
    zmin, zmax = vertices[:,2].min(), vertices[:,2].max()
    lo, hi = zmin + (zmax - zmin) * lower_trim_fraction, zmin + (zmax - zmin) * upper_trim_fraction
    mask = (vertices[:,2] >= lo) & (vertices[:,2] <= hi)
    if mask.sum() < min_vertex_threshold:
        print("  Too few verts after trimming—skipping trimline.")
        return vertices, faces
    new_idx = -np.ones(len(vertices), dtype=int)
    vidx = np.where(mask)[0]
    new_idx[vidx] = np.arange(len(vidx))
    faces = np.array(faces)
    fmask = mask[faces].all(axis=1)
    new_faces = new_idx[faces[fmask]]
    return vertices[mask], new_faces

def process_subject_v11(pre_file, post_file, mirror_flag=False, target_points=30000):
    """
    Runs the full rigid+nonrigid registration + trimming + sampling pipeline for one subject:
    1) Load pre‐rectified (ref) & post‐rectified (tgt) meshes
    2) Optionally mirror the target
    3) Align + sample the reference at coronal plane
    4) Align + smooth + trim + register the target to reference
    5) Sample the final registered shape
    Returns: (ref_vector, tgt_vector, ref_obj, reg_obj)
    """
    ref_obj = AmpObject(pre_file)
    tgt_obj = AmpObject(post_file)
    if mirror_flag:
        tgt_obj = mirror_shape(tgt_obj)
    ref_obj.vert = rotate_to_coronal_plane_v11(ref_obj.vert)
    ref_vec = sample_mesh(ref_obj, target_points)
    # Align target vertically to reference
    rzmin, rzmax = ref_obj.vert[:,2].min(), ref_obj.vert[:,2].max()
    tzmin, tzmax = tgt_obj.vert[:,2].min(), tgt_obj.vert[:,2].max()
    rmid, tmid = (rzmin + rzmax) / 2, (tzmin + tzmax) / 2
    thr, tht = 0.1 * (rzmax - rzmin), 0.1 * (tzmax - tzmin)
    rmid_pts = ref_obj.vert[np.abs(ref_obj.vert[:,2] - rmid) < thr]
    tmid_pts = tgt_obj.vert[np.abs(tgt_obj.vert[:,2] - tmid) < tht]
    rxy = rmid_pts[:, :2].mean(axis=0)
    txy = tmid_pts[:, :2].mean(axis=0)
    translation = np.array([rxy[0] - txy[0], rxy[1] - txy[1], rmid - tmid])
    print(f"  Translating target by {translation}")
    tgt_obj.vert += translation
    tgt_obj.vert = rotate_to_coronal_plane_v11(tgt_obj.vert)
    align(ref_obj, tgt_obj).runICP()
    print("  Rigid ICP done")
    tgt_obj.vert = hc_laplacian_smooth_v11(tgt_obj.vert, tgt_obj.faces, iterations=3)
    print("  HC smoothing done")
    v, f = adaptive_trimline(tgt_obj.vert, tgt_obj.faces)
    if v.shape[0] >= 25000:
        tgt_obj.vert, tgt_obj.faces = v, f
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(tgt_obj.vert)
        mesh.triangles = o3d.utility.Vector3iVector(tgt_obj.faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        tgt_obj.vert = np.asarray(mesh.vertices)
        tgt_obj.faces = np.asarray(mesh.triangles)
        print("  Trimline cleanup done")
    try:
        reg = registration(ref_obj, tgt_obj, steps=5, smooth=1).reg
    except Exception as e:
        print("  Non‐rigid registration failed:", e)
        sys.exit(1)
    print("  Non‐rigid registration done")
    tgt_vec = sample_mesh(reg, target_points)
    return ref_vec, tgt_vec, ref_obj, reg

def collect_subjects(pre_dir, post_dir, mirror_flag=False, target_points=30000):
    """
    Loops over matched pre & post STL files, runs process_subject_v11 for each pair,
    and returns:
      • refs:   np.vstack of all ref‐shape vectors
      • tgts:   np.vstack of all tgt‐shape vectors
      • regs:   list of all registered AmpObject (reg_obj) for clustering
      • ids:    list of subject IDs (extracted from filename)
    """
    pre_files  = sorted(f for f in os.listdir(pre_dir)  if f.lower().endswith('.stl'))
    post_files = sorted(f for f in os.listdir(post_dir) if f.lower().endswith('.stl'))
    if len(pre_files) != len(post_files):
        raise ValueError("Mismatch in pre‐rectified vs post‐rectified STL counts")
    refs, tgts, regs, ids = [], [], [], []
    for p, q in zip(pre_files, post_files):
        # Attempt to extract a numeric subject ID from the filename, e.g. “3D Scan 3.stl” → 3
        pid_match = re.search(r"(\d+)", p)
        if pid_match:
            pid = int(pid_match.group(1))
        else:
            print(f"  Warning: could not parse numeric ID from {p}, skipping.")
            continue
        try:
            pre_path  = os.path.join(pre_dir, p)
            post_path = os.path.join(post_dir, q)
            rv, tv, ro, rg = process_subject_v11(
                pre_path, post_path, mirror_flag, target_points
            )
            refs.append(rv)
            tgts.append(tv)
            regs.append((pid, rg))  # store tuple (subjectID, registeredAmpObject)
            ids.append(pid)
            print(f"Processed subject {pid} → {p}")
        except Exception as e:
            print(f"Error on {p}: {e}")
    return np.vstack(refs), np.vstack(tgts), regs, ids

def build_pca_model(data, n_components=11):
    """
    Standard PCA builder that returns (mean, components, scores, explained_variance_ratio).
    Clips standardized data to +/-2 before PCA.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    clipped = np.clip(scaled, -2, 2)
    k = min(n_components, clipped.shape[0] - 1, clipped.shape[1])
    pca = PCA(n_components=k)
    scores = pca.fit_transform(clipped)
    return scaler.mean_, pca.components_, scores, pca.explained_variance_ratio_

def pca_summary_table(var_model, n_pcs=3):
    """
    Returns a pandas DataFrame summarizing the % variance for the first n_pcs.
    """
    n = min(n_pcs, len(var_model))
    d = {f"PC{i+1} var (%)": [var_model[i] * 100] for i in range(n)}
    d[f"Total var (first {n} PCs) (%)"] = [np.sum(var_model[:n]) * 100]
    return pd.DataFrame(d)

def pca_detailed_table(data, n_components=11):
    """
    Returns a DataFrame listing eigenvalue, variance%, cumulative% for each component.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    clipped = np.clip(scaled, -2, 2)
    k = min(n_components, clipped.shape[0] - 1, clipped.shape[1])
    pca = PCA(n_components=k).fit(clipped)
    ev = pca.explained_variance_
    vr = pca.explained_variance_ratio_
    cv = np.cumsum(vr)
    return pd.DataFrame({
        "Component":    np.arange(1, k + 1),
        "Eigenvalue":   ev,
        "% Variance":   vr * 100,
        "Cumulative %": cv * 100
    })

def generate_synthetic_shapes(mean_shape, components, scores, n_show=5):
    """
    Builds and plots synthetic shapes at +/-2σ along each of the first n_show PCA modes.
    """
    low  = np.percentile(scores,  2.5, axis=0)
    high = np.percentile(scores, 97.5, axis=0)
    shapes = []
    for i in range(components.shape[0]):
        δl = np.zeros(components.shape[0])
        δh = δl.copy()
        δl[i] = low[i]
        δh[i] = high[i]
        shapes.append((
            i + 1,
            mean_shape + δl @ components,
            mean_shape + δh @ components
        ))
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(12, 6))
    for idx, (mode, sl, sh) in enumerate(shapes[:n_show]):
        ax1 = fig.add_subplot(2, n_show, idx + 1, projection='3d')
        pts = sl.reshape(-1, 3)
        ax1.scatter(*pts.T, c="tab:red", s=1, alpha=0.6)
        ax1.set_title(f"Mode {mode} Low")
        ax2 = fig.add_subplot(2, n_show, n_show + idx + 1, projection='3d')
        pts2 = sh.reshape(-1, 3)
        ax2.scatter(*pts2.T, c="tab:blue", s=1, alpha=0.6)
        ax2.set_title(f"Mode {mode} High")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modes_custom_colors.png"), dpi=300)
    plt.close()

def mean_shape_error_mm_for_dataset(data, max_components=11):
    """
    Returns mean shape error (in mm) vs number of PCA components 1..max_components.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    errors = []
    n = data.shape[0]
    max_c = min(max_components, n)
    for k in range(1, max_c + 1):
        pca = PCA(n_components=k)
        sc  = pca.fit_transform(scaled)
        recon_s = pca.inverse_transform(sc)
        recon   = recon_s * scaler.scale_ + scaler.mean_
        errors.append(np.linalg.norm(data - recon, axis=1).mean())
    return np.array(errors)

def plot_mean_shape_error_mm(ref_ds, tgt_ds, comb_ds, max_components=11):
    """
    Plots mean shape error vs number of PCA components for pre, post, and combined,
    using custom colors from PCA_PALETTE.
    """
    mse_r = mean_shape_error_mm_for_dataset(ref_ds, max_components)
    mse_t = mean_shape_error_mm_for_dataset(tgt_ds, max_components)
    mse_c = mean_shape_error_mm_for_dataset(comb_ds, max_components)
    x = np.arange(1, len(mse_r) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, mse_r, marker='o', linestyle='-', color=PCA_PALETTE["pre"], label='Pre-rectified')
    plt.plot(x, mse_t, marker='o', linestyle='-', color=PCA_PALETTE["post"], label='Post-rectified')
    plt.plot(x, mse_c, marker='o', linestyle='-', color=PCA_PALETTE["combined"], label='Combined')
    plt.xlabel("No. of Components")
    plt.ylabel("Mean Shape Error (mm)")
    plt.title("Mean Shape Error vs. Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_shape_error_custom_colors.png"), dpi=300)
    plt.close()

def apply_colormap_with_CMapOut(obj):
    """
    Applies a coolwarm colormap to obj.scalar (e.g. deviation distances) and renders in VTK.
    Returns (min, max) scalar range.
    """
    mn, mx = obj.scalar.min(), obj.scalar.max()
    print(f"Colormap range: {mn:.2f} - {mx:.2f}")
    cmap = plt.get_cmap('coolwarm')(np.linspace(0, 1, 256))
    CMapOut(obj, colors=cmap)
    obj.actor.setScalarRange([mn, mx])
    obj.actor.setShading(False)
    return mn, mx

class EnhancedGUI:
    """
    A helper class to launch an interactive VTK window for a given obj.actor.
    """
    def __init__(self, actor):
        self.actor = actor
        self.vtk_window = None
    def setup(self):
        try:
            self.vtk_window = qtVtkWindow()
            ren = vtk.vtkRenderer()
            self.vtk_window.GetRenderWindow().AddRenderer(ren)
            ren.AddActor(self.actor)
            ren.SetBackground(1, 1, 1)
            ren.ResetCamera()
            self.vtk_window.GetRenderWindow().Render()
            return True
        except Exception as e:
            print("VTK setup failed:", e)
            return False
    def launch(self):
        app = QApplication(sys.argv)
        if not self.setup():
            return
        self.vtk_window.show()
        sys.exit(app.exec_())

# ──────────────────────────────────────────────────────────────────────────────
#          SECTION 3 (UPDATED): 2D PROJECTION → SEGMENTATION → FEATURE EXTRACTION → K‐MEANS
# ──────────────────────────────────────────────────────────────────────────────

from skimage.feature import hog
from skimage import measure
from skimage.filters import threshold_otsu

def generate_deviation_map_2d(reg_obj, ref_obj, lm_dict, image_size=(224, 224), z_slop=1.0, axis=2):
    """
    1) Computes a per‐vertex “deviation” scalar for reg_obj (post‐rectified) by finding
       the nearest‐neighbor distance to ref_obj (pre‐rectified) via KDTree.
    2) Extracts a coronal slab at the iliac‐crest level (z = L-ASIS) from reg_obj, with +/- z_slop.
    3) Projects those slab points onto the XY plane and “paints” a 2D deviation image
       of size image_size (H×W), where each pixel value is the maximum deviation of any
       3D point that falls into that pixel. Pixels with no points remain zero.
    Returns:
      • dev_img: a 2D np.array of shape (H, W) with float-valued deviations >= 0.
      • mask_img: a 2D boolean mask (H, W) indicating where reg_obj had >0 deviation.
    """
    # 1) Build KDTree on ref_obj vertices
    ref_pts = ref_obj.vert                # (N_ref, 3)
    reg_pts = reg_obj.vert                # (N_reg, 3)
    tree    = cKDTree(ref_pts)
    dists, _ = tree.query(reg_pts)        # (N_reg,) distances to nearest pre‐vertex

    # 2) Extract z = section_z(lm, "L-ASIS"); get slab of reg_obj near that z
    z = section_z(lm_dict, "L-ASIS")
    if z is None:
        return None, None
    slab = analyse.create_slices(reg_obj, [z - z_slop, z + z_slop], 200, typ="real_intervals", axis=axis)
    if not (slab and len(slab[0]) > 0):
        return None, None
    pts3d = np.vstack(slab[0])                     # (M, 3) all points in slab
    # We need to know each point’s deviation: match slab pts back to reg_pts indices.
    # Easiest approach: for each slab point, find the closest point index in reg_pts.
    # But that is expensive: instead, use a faster approach: build KDTree on reg_pts -> index of slab points
    reg_tree = cKDTree(reg_pts)
    _, reg_inds = reg_tree.query(pts3d)            # (M,) indices into reg_pts
    dev_vals = dists[reg_inds]                     # (M,) deviation for each slab point

    # 3) Project slab pts (pts3d) onto XY (for coronal, axis=2 -> XY plane).
    pts2d = pts3d[:, [0, 1]]  # (M, 2)
    # Normalize to image coords [0..W-1], [0..H-1]
    x_min, x_max = pts2d[:,0].min(), pts2d[:,0].max()
    y_min, y_max = pts2d[:,1].min(), pts2d[:,1].max()
    if (x_max - x_min) < 1e-3 or (y_max - y_min) < 1e-3:
        return None, None
    H, W = image_size
    x_idx = ((pts2d[:,0] - x_min) / (x_max - x_min) * (W - 1)).astype(int)
    y_idx = ((pts2d[:,1] - y_min) / (y_max - y_min) * (H - 1)).astype(int)
    # Create a blank deviation image; fill each pixel with the maximum dev among points that fall there
    dev_img = np.zeros((H, W), dtype=np.float32)
    for (xi, yi, dv) in zip(x_idx, y_idx, dev_vals):
        # In image space, invert y so larger Y in world maps to lower row index
        row = H - 1 - yi
        col = xi
        if dev_img[row, col] < dv:
            dev_img[row, col] = dv
    mask_img = dev_img > 0
    return dev_img, mask_img

def segment_regions(dev_img, mask_img, min_region_area=20):
    """
    Given:
      • dev_img:   2D float array of deviations
      • mask_img:  2D bool mask where deviation > 0
    This function:
      1) Computes an adaptive threshold via Otsu on dev_img[mask_img] to get binarized map.
      2) Labels connected components in that binary map.
      3) Filters out any small regions (< min_region_area pixels).
    Returns:
      • regions: a list of skimage-regionprops objects, each describing one connected region.
      • labels: labeled image of same size as dev_img
    """
    if dev_img is None or mask_img is None:
        return [], None

    # 1) Use Otsu threshold on the nonzero deviations
    try:
        thresh = threshold_otsu(dev_img[mask_img])
    except ValueError:
        # If dev_img[mask_img] is empty, return no regions
        return [], None
    binary = dev_img >= thresh

    # 2) Label connected regions
    labels = measure.label(binary, connectivity=1)
    props  = measure.regionprops(labels)

    # 3) Filter out small regions
    regions = [r for r in props if r.area >= min_region_area]
    return regions, labels

def extract_region_features(regions, labels, dev_img, image_size=(224, 224), hog_pixels_cell=(16,16), hog_cells_block=(2,2)):
    """
    For each binary region (from skimage.regionprops), compute:
      (a) HOG on that region’s bounding-box cropped and resized to a fixed size
      (b) 7×7 grid location features: for each grid cell, count how many of the region’s pixels fall there
      (c) total pixel count (region.area)
    Returns:
      • feat_list: a list of 1D feature‐vectors, one per region.
      • region_labels: a parallel list of integers indicating the region’s label index.
    """
    H, W = image_size
    feat_list = []
    region_labels = []

    # Build an empty “full‐image” mask for easy location‐count
    full_mask = (labels > 0)

    # Create a 7×7 grid over the entire image
    grid_h = H // 7
    grid_w = W // 7

    for region in regions:
        # (i) Bounding box of this region
        minr, minc, maxr, maxc = region.bbox  # in row/col coords
        # Crop binary mask to region bounding box
        bb_mask = labels[minr:maxr, minc:maxc] == region.label  # bool array

        # Resize the cropped mask to a fixed 64×64 for HOG (or any reasonable size)
        # We will pad or crop to ensure 64×64
        bb_h, bb_w = bb_mask.shape
        target_size = (64, 64)
        # Simple nearest‐neighbor resize via skimage.transform.resize (anti_aliasing=False, preserve_range=True)
        from skimage.transform import resize
        bb_resized = resize(bb_mask.astype(float), target_size,
                            order=0, preserve_range=True, anti_aliasing=False)
        bb_resized = (bb_resized >= 0.5).astype(float)  # back to binary float

        # Compute HOG features for this region crop
        hog_fd = hog(bb_resized,
                     pixels_per_cell=hog_pixels_cell,
                     cells_per_block=hog_cells_block,
                     feature_vector=True)

        # (ii) 7×7 grid location features: count region pixels in each cell of the full‐image grid
        loc_feats = []
        # Extract all pixel coordinates belonging to this region in the full image:
        coords = region.coords  # (N, 2) array of (row, col)
        for gi in range(7):
            for gj in range(7):
                r0 = gi * grid_h
                r1 = (gi + 1) * grid_h if gi < 6 else H
                c0 = gj * grid_w
                c1 = (gj + 1) * grid_w if gj < 6 else W
                # Count how many (row, col) fall in [r0:r1, c0:c1]
                count = np.sum((coords[:,0] >= r0) & (coords[:,0] < r1) &
                               (coords[:,1] >= c0) & (coords[:,1] < c1))
                loc_feats.append(count)

        # (iii) Total pixel count (region.area)
        pixel_count = region.area

        # (iv) Combine features: [HOG dims, loc_feats (49), pixel_count (1)] → total ~ (len(hog_fd)+50)
        feat = np.hstack([hog_fd, loc_feats, pixel_count])
        feat_list.append(feat)
        region_labels.append(region.label)

    return np.array(feat_list), region_labels

def run_region_level_kmeans(reg_objs, ref_objs, lm_dicts, image_size=(224,224), n_clusters=5):
    """
    For each subject (identified by index in reg_objs), do:
      1) generate a 2D deviation map (dev_img, mask_img)
      2) segment regions -> list of region objects + labeled image
      3) extract region features -> (feat_matrix, region_labels)
      4) Keep track of (patientID, regionBoundingBox, regionLabel, featureVector)
    After processing all subjects, stack all region-level features into a single 2D array
    -> run PCA (to 25 dims) -> run KMeans(n_clusters)
    Returns:
      • all_region_info: a list of dicts with keys:
          { 'PatientID', 'RegionLabel', 'BBox', 'Cluster', 'Coords' }
      • kmeans_model: trained KMeans model
      • pca_model: trained PCA model on region features
      • cluster_labels: list of cluster assignments (parallel to all_region_info)
    """
    all_feats = []
    all_info = []
    for (pid, reg_obj), ref_obj in zip(reg_objs, ref_objs):
        lm = pre_lm.get(pid, None)
        if lm is None:
            continue

        # 1) Generate deviation map for this subject
        dev_img, mask_img = generate_deviation_map_2d(reg_obj, ref_obj, lm, image_size=image_size)
        if dev_img is None:
            continue

        # 2) Segment connected regions in this dev-mask
        regions, labels_img = segment_regions(dev_img, mask_img, min_region_area=20)
        if not regions:
            continue

        # 3) Extract features from each region
        feats, region_labels = extract_region_features(regions, labels_img, dev_img, image_size=image_size)
        if feats.size == 0:
            continue

        # 4) Store each region’s info
        for idx, region in enumerate(regions):
            info = {
                "PatientID": pid,
                "RegionLabel": region_labels[idx],
                "BBox": region.bbox,   # (min_row, min_col, max_row, max_col)
                "Coords": region.coords  # array of pixel coords in full image
            }
            all_info.append(info)
        all_feats.append(feats)

    if not all_feats:
        return [], None, None, []

    # Stack all region features across all subjects
    all_feats_mat = np.vstack(all_feats)   # shape (TotalRegions, FeatureDim)

    # 5) PCA dimensionality reduction to 25 components (per paper)
    n_components = min(25, all_feats_mat.shape[1], all_feats_mat.shape[0]-1)
    pca = PCA(n_components=n_components)
    feats_pca = pca.fit_transform(all_feats_mat)

    # 6) KMeans on PCA-reduced features
    km = KMeans(n_clusters=n_clusters, random_state=0)
    region_clusters = km.fit_predict(feats_pca)  # (TotalRegions,)

    # 7) Assign cluster labels back into all_info
    for i, cl in enumerate(region_clusters):
        all_info[i]["Cluster"] = int(cl)

    return all_info, km, pca, region_clusters

def evaluate_region_clustering(all_info, labels_df, output_prefix):
    """
    Build a DataFrame with one row per region, containing:
      PatientID, Cluster, BinaryLabel, MulticlassLabel
    Then compute and save:
      • Confusion matrix: Binary vs Cluster
      • Confusion matrix: Multiclass vs Cluster
    """
    # Build DataFrame
    records = []
    for info in all_info:
        pid = info["PatientID"]
        rec = {
            "PatientID": pid,
            "Cluster": info["Cluster"]
        }
        # Lookup labels
        lbl_row = labels_df[labels_df["PatientID"] == pid]
        if lbl_row.empty:
            rec["BinaryLabel"] = None
            rec["MultiLabel"]  = None
        else:
            rec["BinaryLabel"] = lbl_row["Thoracic_Correction_Binary"].values[0]
            rec["MultiLabel"]  = lbl_row["Dominant_Correction_Region"].values[0]
        records.append(rec)

    df_regions = pd.DataFrame(records).dropna(subset=["BinaryLabel", "MultiLabel"])
    # Binary confusion
    bin_cm, bin_png, _, _ = evaluate_clusters_vs_labels(
        df_regions["Cluster"].values,
        df_regions["PatientID"].values,
        labels_df.rename(columns={"Thoracic_Correction_Binary": "Label"}),
        "Thoracic_Correction_Binary",
        output_prefix
    )
    # Multiclass confusion
    multi_cm, multi_png, _, _ = evaluate_clusters_vs_labels(
        df_regions["Cluster"].values,
        df_regions["PatientID"].values,
        labels_df.rename(columns={"Dominant_Correction_Region": "Label"}),
        "Dominant_Correction_Region",
        output_prefix
    )
    # Also save the raw region-level assignment CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{output_prefix}_region_assignments.csv")
    df_regions.to_csv(out_csv, index=False)
    print(f"→ Saved region assignments to {out_csv}")
    return bin_cm, bin_png, multi_cm, multi_png

# ──────────────────────────────────────────────────────────────────────────────
#       SECTION 4: METADATA + CLUSTER SECTION (TABLE 3 + Clustering Eval)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_clinical_metadata(new_summary_csv_path):
    """
    Loads the newly uploaded CSV (brace_summary_statistics_table.csv) containing per‐patient Cobb 
    and related metadata. Then:
      1) Copies it into the OUTPUT_DIR as 'brace_summary_statistics_table3.csv'.
      2) Extracts (PatientID, Thoracic Diff, Lumbar Diff, Thoraco-Lumbar Diff, LenkeType).
      3) Builds:
         - Binary “Thoracic_Correction_Binary”: “Improved” if |Thoracic_Diff| >= 10, else “Not Improved”.
         - Multiclass “Dominant_Correction_Region”: whichever of Thoracic/Lumbar/Thoraco‐Lumbar has the largest |Δ|.
      4) Saves a processed metadata CSV (with all combined columns) as 
         'brace_metadata_processed.csv' in OUTPUT_DIR.

    Returns:
      • metadata_df (DataFrame): The filtered per‐patient data with Cobb measures and new labels.
      • labels_df   (DataFrame): [PatientID, Thoracic_Correction_Binary, Dominant_Correction_Region]
    """
    # 1) Load the new summary table you provided
    df = pd.read_csv(new_summary_csv_path)
    # The original CSV has an “Unnamed: 0” column—drop it if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2) Rename columns for consistency with the rest of the pipeline
    df = df.rename(columns={
        "Patient's Number (New System)":       "PatientID",
        "Lenke Curve Type (1,2,3,4,5,6)":       "LenkeType",
        "Thoracic Cobb Angle (Before Brace)":   "Thoracic_InBrace",
        "Lumbar Cobb Angle (Before Brace)":     "Lumbar_InBrace",
        "Thoraco Lumbar Cobb Angle (Before Brace)": "ThoracoLumbar_InBrace",
        "Thoracic Cobb Angle (In-Brace)":       "Thoracic_InBrace",       # if this column appears
        "Lumbar Cobb Angle (In-Brace)":         "Lumbar_InBrace",         # (sometimes the “In-Brace” columns are separate)
        "Thoraco Lumbar Cobb Angle (In-Brace)": "ThoracoLumbar_InBrace",  # adjust if needed
        "Thoracic Cobb Angle Difference":       "Thoracic_Diff",
        "Lumbar Cobb Angle Difference":         "Lumbar_Diff",
        "Thoraco Lumbar Cobb Angle Difference": "ThoracoLumbar_Diff"
    })

    # 3) Copy this new CSV into OUTPUT_DIR as "brace_summary_statistics_table.csv"
    summary_dest = os.path.join(OUTPUT_DIR, "brace_summary_statistics_table.csv")
    df.to_csv(summary_dest, index=False)
    print(f"→ Copied new summary table to {summary_dest}")

    # 4) Build the two new labels based on the “Difference” columns:
    #    (a) Binary label: “Improved” if |Thoracic_Diff| >= 10°, else “Not Improved”
    df["Thoracic_Correction_Binary"] = df["Thoracic_Diff"].apply(
        lambda x: "Improved" if pd.notna(x) and abs(x) >= 10 else "Not Improved"
    )

    #    (b) Multiclass label: whichever of Thoracic/Lumbar/Thoraco‐Lumbar has the largest |Δ|
    def classify_multiclass(row):
        angles = {
            "Thoracic":      abs(row["Thoracic_Diff"])       if pd.notna(row["Thoracic_Diff"])       else 0,
            "Lumbar":        abs(row["Lumbar_Diff"])         if pd.notna(row["Lumbar_Diff"])         else 0,
            "Thoraco-Lumbar": abs(row["ThoracoLumbar_Diff"]) if pd.notna(row["ThoracoLumbar_Diff"]) else 0
        }
        # Return the key corresponding to the maximum absolute correction
        return max(angles, key=angles.get)

    df["Dominant_Correction_Region"] = df.apply(classify_multiclass, axis=1)

    # 5) Select and reorder the relevant columns for metadata & labels
    metadata_cols = [
        "PatientID",
        "LenkeType",
        "Thoracic_InBrace",
        "Lumbar_InBrace",
        "ThoracoLumbar_InBrace",
        "Thoracic_Diff",
        "Lumbar_Diff",
        "ThoracoLumbar_Diff",
        "Thoracic_Correction_Binary",
        "Dominant_Correction_Region"
    ]
    # Some columns (e.g. Lumbar_InBrace or ThoracoLumbar_InBrace) may be missing or NaN for certain patients;
    # this DataFrame will carry forward whatever is in the CSV.
    metadata = df[metadata_cols].copy()

    # 6) Save the processed metadata that downstream steps will use
    metadata_dest = os.path.join(OUTPUT_DIR, "brace_metadata_processed.csv")
    metadata.to_csv(metadata_dest, index=False)
    print(f"→ Saved processed metadata to {metadata_dest}")

    # 7) Build labels_df for clustering evaluation (just the two new keys)
    labels_df = metadata[["PatientID", "Thoracic_Correction_Binary", "Dominant_Correction_Region"]].copy()

    return metadata, labels_df

# ──────────────────────────────────────────────────────────────────────────────
#                  SECTION 5: MAIN PIPELINE (ALP + PCA + CLUSTERING)
# ──────────────────────────────────────────────────────────────────────────────

def run_combined_pipeline():
    start = time.time()

    # ─────── ALP PIPELINE ───────
    print("\n===== RUNNING ALP PIPELINE =====")
    run_alp_pipeline()

    # ─────── CLINICAL METADATA ───────
    print("\n===== PREPARING CLINICAL METADATA =====")
    new_summary_csv = r"C:\Users\jm1321\OneDrive - Imperial College London\_FYP\data_including_aop_files\Data_with_landmark_1\brace_summary_statistics_table.csv"
    metadata_df, labels_df = prepare_clinical_metadata(new_summary_csv)

    # ─────── COLLECT SHAPES & PCA (PRE, POST, COMBINED) ───────
    print("\n===== COLLECTING SHAPES & PCA (PRE, POST, COMBINED) =====")
    ref_ds, tgt_ds, regs, ids = collect_subjects(
        PRE_STL_DIR,
        POST_STL_DIR,
        mirror_flag=False,
        target_points=30000
    )
    print(f"→ Ref shape data: {ref_ds.shape}")
    print(f"→ Tgt shape data: {tgt_ds.shape}")

    # Save raw datasets for future reference
    np.save(os.path.join(OUTPUT_DIR, "ref_dataset.npy"), ref_ds)
    np.save(os.path.join(OUTPUT_DIR, "target_dataset.npy"), tgt_ds)

    # Pre-rectified PCA
    print("\n Pre-rectified PCA:")
    m_r, c_r, s_r, v_r = build_pca_model(ref_ds)
    print(pca_summary_table(v_r))
    print(pca_detailed_table(ref_ds))

    print("\n Generating synthetic shapes (pre-rectified)...")
    generate_synthetic_shapes(m_r, c_r, s_r, n_show=5)

    # Post-rectified PCA
    print("\n Post-rectified PCA:")
    m_t, c_t, s_t, v_t = build_pca_model(tgt_ds)
    print(pca_summary_table(v_t))
    print(pca_detailed_table(tgt_ds))

    # Combined PCA
    print("\n Combined PCA:")
    if ref_ds.shape[0] != tgt_ds.shape[0]:
        raise ValueError("Mismatch in pre & post dataset counts for combined PCA")
    comb_ds = np.hstack((ref_ds, tgt_ds))
    np.save(os.path.join(OUTPUT_DIR, "combined_dataset.npy"), comb_ds)
    m_c, c_c, s_c, v_c = build_pca_model(comb_ds)
    print(pca_summary_table(v_c))
    print(pca_detailed_table(comb_ds))

    modes = [np.searchsorted(np.cumsum(v), 0.95) + 1 for v in (v_r, v_t, v_c)]
    print(f" Modes @95% var (pre, post, comb): {modes}")

    loo_err = leave_one_out_test(comb_ds, n_components=modes[2])
    print(f" LOO error (combined): {loo_err.mean():.4f}")

    print("\n Plotting cumulative variance & LOO errors (Figure 4)...")
    plot_cumulative_variance_and_loo_errors(v_r, v_t, v_c, ref_ds, tgt_ds, comb_ds)

    print("\n Auto-PCA (combined) & 3D scatter (Figure 4g)...")
    pca_auto, pca_data, ev_auto, scaler_auto = build_pca_model_auto(comb_ds)
    if pca_data.shape[1] >= 3:
        plot_pca_3d_scatter(pca_data, ids, n_show=len(ids), labels=ids)

    print("\n Plotting mean shape error vs components (Figure 4d-f)...")
    plot_mean_shape_error_mm(ref_ds, tgt_ds, comb_ds)

    # ─────── CSA (DEVIATION-BASED) ───────
    # Choose the first available registered object as demo
    demo_ref_obj = None
    demo_reg_obj = None
    for pid, reg_obj in regs:
        demo_ref_obj = AmpObject(os.path.join(PRE_STL_DIR, f"3D Scan {pid}.stl"))
        demo_ref_obj = align_to_pelvis(demo_ref_obj, pre_lm[pid])
        demo_reg_obj = reg_obj
        break

    if demo_ref_obj is not None and demo_reg_obj is not None:
        print("\n===== CSA & VTK VISUALIZATION FOR DEMO SUBJECT =====")
        # Compute per-vertex deviation = nearest-neighbor distance
        ref_pts = demo_ref_obj.vert               # (N_ref, 3)
        reg_pts = demo_reg_obj.vert               # (N_reg, 3)
        tree    = cKDTree(ref_pts)
        dists, _ = tree.query(reg_pts)            # (N_reg,)
        dev = dists                               # per-vertex deviation
        demo_reg_obj.scalar = dev
        apply_colormap_with_CMapOut(demo_reg_obj)

        try:
            zmin, zmax = demo_reg_obj.vert[:,2].min(), demo_reg_obj.vert[:,2].max()
            sl = analyse.create_slices(demo_reg_obj, [zmin, zmax], 50, typ="real_intervals", axis=2)
            csa = analyse.calc_csa(sl)
            heights = np.arange(zmin, zmin + len(csa)*50, 50)
            csa_csv = os.path.join(OUTPUT_DIR, "cross_sectional_areas.csv")
            with open(csa_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Height (mm)", "Area (mm²)"])
                w.writerows(zip(heights, csa))
            print(f"→ CSA saved to {csa_csv}")
            plt.figure(figsize=(8, 5))
            plt.plot(heights, csa, marker='o', linestyle='-', color="tab:green")
            plt.xlabel("Height (mm)")
            plt.ylabel("Area (mm²)")
            plt.title("Cross-Sectional Areas")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "cross_sectional_areas_custom_color.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(" CSA analysis failed:", e)

        print(" Interactive VTK viewer skipped (set to False by default).")
    else:
        print(" No demo subject available for CSA.")

    # ─────── REGION-LEVEL CLUSTERING ───────
    print("\n===== RUNNING REGION-LEVEL CLUSTERING =====")
    # Build a parallel list of pre-aligned reference objects
    ref_objs = []
    for pid, _ in regs:
        ro = AmpObject(os.path.join(PRE_STL_DIR, f"3D Scan {pid}.stl"))
        ro = align_to_pelvis(ro, pre_lm[pid])
        ref_objs.append(ro)

    # regs: list of (PatientID, registeredAmpObject)
    # ref_objs: list of AmpObject (pre-aligned), same order
    all_info, km_model, pca_model, region_clusters = run_region_level_kmeans(
        regs,       # [(PatientID, reg_obj), …]
        ref_objs,   # [pre_aligned_obj, …]
        pre_lm,     # to look up section_z(lm, "L-ASIS")
        image_size=(224, 224),
        n_clusters=5
    )

    if all_info:
        print(f"→ Extracted {len(all_info)} total regions across all subjects.")
        # Evaluate clustering vs. clinical labels
        evaluate_region_clustering(all_info, labels_df, output_prefix="Clustering")
    else:
        print("→ No regions found; skipping clustering evaluation.")

    elapsed = time.time() - start
    print(f"\n=== FULL PIPELINE COMPLETE in {elapsed:.1f}s ===")

# ──────────────────────────────────────────────────────────────────────────────
#                      PLOT CUMULATIVE VAR & LOO ERRORS (Figure 4)
# ──────────────────────────────────────────────────────────────────────────────

def plot_cumulative_variance_and_loo_errors(var_r, var_t, var_c, ref_ds, tgt_ds, comb_ds, max_components=11):
    """
    Creates a 2×3 grid:
      Top row: cumulative variance plots for pre, post, combined
      Bottom row: leave-one-out reconstruction error vs components for pre, post, combined
    Uses custom colors from PCA_PALETTE and labels each subplot (a)–(f).
    """
    n = min(ref_ds.shape[0], tgt_ds.shape[0], comb_ds.shape[0])
    max_c = min(max_components, n - 1)
    x = np.arange(1, max_c + 1)
    cum_r = np.cumsum(var_r[:max_c])
    cum_t = np.cumsum(var_t[:max_c])
    cum_c = np.cumsum(var_c[:max_c])
    err_r = loo_error_for_n_components(ref_ds, max_c)
    err_t = loo_error_for_n_components(tgt_ds, max_c)
    err_c = loo_error_for_n_components(comb_ds, max_c)

    # figure & axes white
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # Top row: Normalised Cumulative Variance
    for idx, (ax, cum, key, title) in enumerate(zip(
            axes[0],
            [cum_r, cum_t, cum_c],
            ["pre", "post", "combined"],
            ["Pre-rectified", "Post-rectified", "Combined"]
        )):
        ax.set_facecolor('white')
        ax.plot(x, cum, marker='o', linestyle='-', color=PCA_PALETTE[key], linewidth=2)
        ax.set_title(f"{title} Model")
        ax.set_xlabel("No. of Components")
        ax.set_ylabel("Normalised Cumulative Variance")
        ax.set_ylim([0, 1.05])
        ax.axhline(0.95, linestyle='--', color='k')
        ax.grid(True)
        ax.text(0.02, 0.99, subplot_labels[idx],
                transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

    # Bottom row: Normalised LOO Reconstruction Error
    all_err = np.concatenate([err_r, err_t, err_c])
    mn, mx = all_err.min(), all_err.max()

    for idx, (ax, errs, key, title) in enumerate(zip(
            axes[1],
            [err_r, err_t, err_c],
            ["pre", "post", "combined"],
            ["Pre-rectified", "Post-rectified", "Combined"]
        )):
        ax.set_facecolor('white')
        ax.scatter(x, errs, c=PCA_PALETTE[key], marker='x')
        coeffs = np.polyfit(x, errs, 1)
        ax.plot(x, np.poly1d(coeffs)(x), linestyle='--', color=PCA_PALETTE[key])
        ax.set_title(f"{title} LOO Error")
        ax.set_xlabel("No. of Components")
        ax.set_ylabel("Normalised Reconstruction Error")
        ax.set_ylim([mn * 0.95, mx * 1.05])
        ax.grid(True)
        ax.text(0.02, 0.95, subplot_labels[idx + 3],
                transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure4_variance_loo_custom_colors.png"),
                dpi=300, facecolor='white')
    plt.close()
    print("Saved Figure4_variance_loo_custom_colors.png")


# ──────────────────────────────────────────────────────────────────────────────
#                                     MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_combined_pipeline()
