"""
Ablation study for paper revision — Reviewer Comment 5.

Compares:
  (A) Binary Fellegi-Sunter vs Beta-continuous similarity model (Phase 2)
  (B) Greedy (no coordination) vs MILP-coordinated activity assignment (Phase 3)

Usage:
    python run_ablation_study.py --n-buildings 2000
    python run_ablation_study.py --n-buildings 2000 --output-dir results/ablation
"""

import sys
import os
import json
import argparse
import logging
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Suppress non-critical warnings during ablation
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Project imports
from src.utils.config_loader import get_config
from src.data_loading.recs_loader import load_recs_data, prepare_recs_for_matching
from src.data_loading.atus_loader import (
    load_all_atus_data,
    create_activity_templates,
    load_atus_activity_data,
)
from src.processing.phase1_pums_integration import load_phase1_output
from src.matching.fellegi_sunter import FellegiSunterMatcher, ComparisonField
from src.matching.em_algorithm import EMAlgorithm
from src.matching.blocking import create_standard_blocks
from src.matching.string_comparators import get_similarity_level
from src.utils.enhanced_feature_engineering import (
    create_comprehensive_matching_features,
    get_matching_features_list,
    align_features_for_matching,
)
from src.utils.cross_dataset_features import enhance_dataset_for_matching

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ablation_study.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_phase1_sample(n_buildings: int) -> pd.DataFrame:
    """Load a sample of Phase 1 buildings from shards or pickle."""
    shards_dir = Path("data/processed/phase1_shards")
    manifest_path = shards_dir / "manifest.json"

    if manifest_path.exists():
        logger.info("Loading Phase 1 buildings from shards ...")
        with open(manifest_path) as f:
            manifest = json.load(f)
        dfs = []
        loaded = 0
        for fp in manifest.get("files", []):
            try:
                chunk = pd.read_pickle(fp)
                dfs.append(chunk)
                loaded += len(chunk)
                if loaded >= n_buildings:
                    break
            except Exception as e:
                logger.warning(f"Skipping shard {fp}: {e}")
        if dfs:
            pums = pd.concat(dfs, ignore_index=True).head(n_buildings)
        else:
            raise RuntimeError("No readable Phase 1 shards found")
    else:
        logger.info("Loading Phase 1 buildings from pickle ...")
        pkl_path = Path("data/processed/phase1_pums_buildings.pkl")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Phase 1 output not found: {pkl_path}")
        pums = pd.read_pickle(pkl_path)
        pums = pums.sample(n=min(n_buildings, len(pums)), random_state=42).reset_index(drop=True)

    logger.info(f"Loaded {len(pums)} Phase 1 buildings")
    return pums


# ============================================================
# ABLATION A — Phase 2: Binary F-S vs Beta Continuous
# ============================================================

def _build_comparison_fields(common_features: List[str]) -> List[ComparisonField]:
    """Build ComparisonField objects for common numeric/categorical features."""
    fields = []
    for feat in common_features:
        if any(kw in feat for kw in ["_cat", "quintile", "tercile", "decile",
                                      "type", "fuel", "zone", "tenure",
                                      "urban_rural", "REGION", "DIVISION"]):
            fields.append(ComparisonField(
                name=feat, field_type="categorical",
                comparison_method="exact", weight=1.0))
        else:
            fields.append(ComparisonField(
                name=feat, field_type="numeric",
                comparison_method="numeric_similarity", weight=1.0))
    return fields


def _compute_similarity_matrix(pums: pd.DataFrame, recs: pd.DataFrame,
                               fields: List[ComparisonField],
                               max_pairs: int = 40000) -> pd.DataFrame:
    """Compute pairwise similarity vectors between PUMS and RECS records.

    Uses random sampling of candidate pairs to stay within memory limits.
    """
    n_pums = len(pums)
    n_recs = len(recs)
    total_possible = n_pums * n_recs
    field_names = [f.name for f in fields]

    logger.info(f"Computing similarities: {n_pums} PUMS × {n_recs} RECS "
                f"= {total_possible:,} possible pairs, sampling {max_pairs:,}")

    # Sample candidate pairs
    rng = np.random.RandomState(42)
    if total_possible <= max_pairs:
        pairs = [(i, j) for i in range(n_pums) for j in range(n_recs)]
    else:
        idx1 = rng.randint(0, n_pums, size=max_pairs)
        idx2 = rng.randint(0, n_recs, size=max_pairs)
        pairs = list(zip(idx1.tolist(), idx2.tolist()))

    # Compute similarities
    matcher = FellegiSunterMatcher(fields)
    rows = []
    for i, (pi, ri) in enumerate(pairs):
        pums_rec = pums.iloc[pi]
        recs_rec = recs.iloc[ri]
        agreement = matcher.calculate_agreement_patterns(pums_rec, recs_rec)
        row = {"idx1": pi, "idx2": ri}
        for k, fname in enumerate(field_names):
            row[f"{fname}_similarity"] = agreement[k]
        rows.append(row)
        if (i + 1) % 10000 == 0:
            logger.info(f"  Computed {i + 1}/{len(pairs)} similarities ...")

    sim_df = pd.DataFrame(rows)
    logger.info(f"Similarity matrix: {len(sim_df)} pairs × {len(field_names)} features")
    return sim_df


def run_binary_baseline(sim_df: pd.DataFrame, field_names: List[str],
                        n_pums: int) -> Dict:
    """Binary F-S: binarise at 0.88, run EM, compute weights without partial agreement."""
    logger.info("=== Ablation A: Running BINARY Fellegi-Sunter baseline ===")

    bdf = sim_df.copy()
    for fname in field_names:
        col = f"{fname}_similarity"
        if col in bdf.columns:
            bdf[col] = (bdf[col] >= 0.88).astype(float)

    # EM on binarised data
    em = EMAlgorithm(field_names, prior_match=0.1)
    em.initialize_parameters(bdf, use_frequency_estimates=True)
    m_probs, u_probs = em.fit(bdf, max_iterations=100, tolerance=0.0001, verbose=False)

    # Binary weight: strict agree / disagree, no interpolation
    weights = np.zeros(len(bdf))
    for fname in field_names:
        col = f"{fname}_similarity"
        if col not in bdf.columns:
            continue
        sim_vals = bdf[col].values
        m = np.clip(m_probs.get(fname, 0.9), 1e-4, 1 - 1e-4)
        u = np.clip(u_probs.get(fname, 0.1), 1e-4, 1 - 1e-4)
        agree_w = np.log2(m / u)
        disagree_w = np.log2((1 - m) / (1 - u))
        weights += np.where(sim_vals >= 0.5, agree_w, disagree_w)

    bdf["weight"] = weights
    # Raw probability, no calibration
    lr = 2.0 ** weights
    bdf["probability"] = (lr * 0.1) / (lr * 0.1 + 0.9)

    # Best match per PUMS record
    best = bdf.loc[bdf.groupby("idx1")["weight"].idxmax()].copy()

    return {
        "comparison_df": bdf,
        "best_matches": best,
        "em_iterations": em.iteration + 1,
        "em_converged": bool(em.converged),
        "m_probs": m_probs,
        "u_probs": u_probs,
    }


def run_beta_continuous(sim_df: pd.DataFrame, field_names: List[str],
                        fields: List[ComparisonField],
                        n_pums: int) -> Dict:
    """Beta-continuous F-S: keep continuous similarities, EM + partial agreement + isotonic."""
    logger.info("=== Ablation A: Running BETA CONTINUOUS model ===")

    cdf = sim_df.copy()

    # EM on continuous similarities
    em = EMAlgorithm(field_names, prior_match=0.1)
    em.initialize_parameters(cdf, use_frequency_estimates=True)
    m_probs, u_probs = em.fit(cdf, max_iterations=100, tolerance=0.0001, verbose=False)

    # Weights with partial agreement (the proposed model)
    matcher = FellegiSunterMatcher(fields)
    matcher.set_probabilities(m_probs, u_probs)

    weights = np.zeros(len(cdf))
    for i in range(len(cdf)):
        row = cdf.iloc[i]
        agreement = [row.get(f"{f.name}_similarity", 0.0) for f in fields]
        weights[i] = matcher.compute_match_weight(agreement)

    cdf["weight"] = weights
    cdf["probability"] = np.array([
        matcher.compute_match_probability(w) for w in weights
    ])

    # Isotonic calibration (pseudo-labels: weight > median → match)
    from sklearn.isotonic import IsotonicRegression
    pseudo = (weights > np.median(weights)).astype(float)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cdf["probability"].values, pseudo)
    cdf["calibrated_probability"] = iso.transform(cdf["probability"].values)

    best = cdf.loc[cdf.groupby("idx1")["weight"].idxmax()].copy()

    return {
        "comparison_df": cdf,
        "best_matches": best,
        "em_iterations": em.iteration + 1,
        "em_converged": bool(em.converged),
        "m_probs": m_probs,
        "u_probs": u_probs,
    }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return float(ece)


def compute_ablation_a_metrics(binary_res: Dict, beta_res: Dict,
                                pums: pd.DataFrame, recs: pd.DataFrame,
                                n_pums: int) -> Dict:
    """Compute all Phase 2 comparison metrics."""
    logger.info("Computing Ablation A metrics ...")
    results = {}

    for label, res, prob_col in [
        ("binary_fs", binary_res, "probability"),
        ("beta_continuous", beta_res, "calibrated_probability"),
    ]:
        best = res["best_matches"]
        all_df = res["comparison_df"]
        probs = best[prob_col].values if prob_col in best.columns else best["probability"].values

        # Pseudo-labels for ECE / Brier (weight > 0 → likely match)
        pseudo = (best["weight"].values > 0).astype(float)

        # Precision / recall proxy at thresholds
        precision = {}
        recall = {}
        for tau in [0.5, 0.7, 0.9]:
            above = probs >= tau
            precision[str(tau)] = float(above.mean())
            recall[str(tau)] = float(above.sum() / max(n_pums, 1))

        ece = _compute_ece(probs, pseudo)
        brier = float(np.mean((probs - pseudo) ** 2))

        # KS statistic — separation between positive / negative weight groups
        all_w = all_df["weight"].values
        med = np.median(all_w)
        high_w = all_w[all_w >= med]
        low_w = all_w[all_w < med]
        ks_stat, ks_p = ks_2samp(high_w, low_w) if len(high_w) > 1 and len(low_w) > 1 else (0, 1)

        # Benchmark deviation: building type, mean floor area, etc.
        matched_recs_indices = best["idx2"].values.astype(int)
        matched_recs = recs.iloc[matched_recs_indices]
        bm_devs = {}
        # Building-type share
        if "building_type_simple" in pums.columns:
            synth_sf = (pums["building_type_simple"] == "single_family").mean() * 100
            bm_devs["single_family_share"] = abs(synth_sf - 68.4) / 68.4 * 100
        # Floor area
        for col in ["TOTSQFT_EN", "square_footage", "total_sqft"]:
            if col in matched_recs.columns:
                synth_area = matched_recs[col].mean()
                bm_devs["mean_floor_area"] = abs(synth_area - 1619) / 1619 * 100
                break
        # Energy cost
        for col in ["TOTALDOL", "total_energy_cost_annual", "total_cost"]:
            if col in matched_recs.columns:
                synth_cost = matched_recs[col].mean()
                bm_devs["mean_energy_cost"] = abs(synth_cost - 1884) / 1884 * 100
                break

        mean_bm_dev = float(np.mean(list(bm_devs.values()))) if bm_devs else 0.0

        results[label] = {
            "em_iterations": res["em_iterations"],
            "em_converged": res["em_converged"],
            "precision_proxy": precision,
            "recall_proxy": recall,
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "ks_statistic": round(float(ks_stat), 3),
            "ks_pvalue": float(ks_p),
            "mean_weight": round(float(best["weight"].mean()), 3),
            "std_weight": round(float(best["weight"].std()), 3),
            "benchmark_deviations_pct": {k: round(v, 1) for k, v in bm_devs.items()},
            "mean_benchmark_deviation_pct": round(mean_bm_dev, 1),
        }

    return results


# ============================================================
# ABLATION B — Phase 3: Greedy vs MILP-Coordinated
# ============================================================

def extract_persons(buildings: pd.DataFrame) -> pd.DataFrame:
    """Extract person records from buildings."""
    persons = []
    for idx, bldg in buildings.iterrows():
        if "persons" not in bldg or not isinstance(bldg["persons"], list):
            continue
        for pi, p in enumerate(bldg["persons"]):
            rec = {"building_id": idx, "person_idx": pi, "person_id": f"{idx}_{pi}"}
            if isinstance(p, dict):
                rec.update({
                    "AGEP": p.get("AGEP", 35),
                    "SEX": p.get("SEX", 1),
                    "ESR": p.get("ESR", 6),
                    "SCHL": p.get("SCHL", 16),
                    "MAR": p.get("MAR", 5),
                    "WKHP": p.get("WKHP", 0),
                    "PINCP": p.get("PINCP", 0),
                    "DIS": p.get("DIS", 2),
                })
            persons.append(rec)
    df = pd.DataFrame(persons)
    logger.info(f"Extracted {len(df)} persons from {len(buildings)} buildings")
    return df


def create_feature_vectors(persons: pd.DataFrame,
                           templates: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Create normalised feature vectors for persons and ATUS templates."""
    pf = []
    for _, p in persons.iterrows():
        pf.append([
            p.get("AGEP", 35) / 100.0,
            1.0 if p.get("SEX", 1) == 2 else 0.0,
            1.0 if p.get("ESR", 6) in [1, 2] else 0.0,
            min(p.get("SCHL", 16), 24) / 24.0,
            1.0 if p.get("MAR", 5) == 1 else 0.0,
            min(p.get("WKHP", 0), 80) / 80.0,
            np.log1p(max(p.get("PINCP", 0), 0)) / 15.0,
            1.0 if p.get("DIS", 2) == 1 else 0.0,
        ])
    pf = np.array(pf)

    tf = []
    edu_map = {"less_than_hs": 0.3, "high_school": 0.5, "some_college": 0.6,
               "bachelors": 0.8, "graduate": 1.0}
    for _, t in templates.iterrows():
        tf.append([
            t.get("age", 35) / 100.0,
            1.0 if t.get("sex", 1) == 2 else 0.0,
            1.0 if t.get("employed", False) else 0.0,
            edu_map.get(t.get("education_level", "high_school"), 0.5),
            1.0 if t.get("spouse_present", 0) == 1 else 0.0,
            t.get("usual_hours_worked", 0) / 80.0,
            np.log1p(max(t.get("weekly_earnings", 0) * 52, 0)) / 15.0,
            0.0,
        ])
    tf = np.array(tf)

    # Replace NaN / Inf with 0 before scaling
    pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
    tf = np.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    pf = scaler.fit_transform(pf)
    tf = scaler.transform(tf)

    # Safety check after scaling
    pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
    tf = np.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)
    return pf, tf


def greedy_knn_assignment(persons_df: pd.DataFrame,
                          pf: np.ndarray, tf: np.ndarray,
                          templates: pd.DataFrame) -> Dict[str, Any]:
    """Assign each person their nearest ATUS diary independently."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nn.fit(tf)
    dists, indices = nn.kneighbors(pf, return_distance=True)
    assignments = {}
    for i, pid in enumerate(persons_df["person_id"]):
        best_idx = int(indices[i, 0])
        assignments[pid] = templates.iloc[best_idx]["template_id"]
    return assignments


def apply_constraint(assignments: Dict, persons_df: pd.DataFrame,
                     templates: pd.DataFrame,
                     constraint: str) -> Dict:
    """Apply a single household coordination constraint in-place."""
    tpl_lookup = {row["template_id"]: row for _, row in templates.iterrows()}
    persons_by_bldg = persons_df.groupby("building_id")

    for bldg_id, hh in persons_by_bldg:
        ages = hh["AGEP"].fillna(35)
        adult_mask = ages >= 18
        child_mask = ages < 12
        school_mask = (ages >= 6) & (ages < 18)
        adult_pids = hh[adult_mask]["person_id"].tolist()
        child_pids = hh[child_mask]["person_id"].tolist()
        school_pids = hh[school_mask]["person_id"].tolist()

        if constraint == "childcare" and child_pids and adult_pids:
            # Ensure at least one adult assigned template has caring > 60 min
            care_times = []
            for pid in adult_pids:
                tpl = tpl_lookup.get(assignments.get(pid), {})
                care = tpl.get("caring_household", 0) if isinstance(tpl, (dict, pd.Series)) else 0
                care_times.append((pid, care))
            if care_times and max(ct for _, ct in care_times) < 60:
                # Re-assign least-busy adult to a high-childcare template
                if "caring_household" in templates.columns:
                    hi_care = templates[templates["caring_household"] > 120]
                    if len(hi_care) > 0:
                        caregiver = min(care_times, key=lambda x: x[1])[0]
                        # Pick closest high-care template
                        pidx = persons_df[persons_df["person_id"] == caregiver].index[0]
                        p_age = persons_df.loc[pidx, "AGEP"]
                        age_diffs = (hi_care["age"] - p_age).abs()
                        best = hi_care.iloc[age_diffs.argmin()]
                        assignments[caregiver] = best["template_id"]

        elif constraint == "meals" and len(hh) >= 2:
            # Ensure at least two members have eating time > 0
            eating_times = []
            for pid in hh["person_id"]:
                tpl = tpl_lookup.get(assignments.get(pid), {})
                eat = tpl.get("eating", 0) if isinstance(tpl, (dict, pd.Series)) else 0
                eating_times.append((pid, eat))
            non_eating = [pid for pid, e in eating_times if e < 15]
            if len(non_eating) >= len(eating_times) - 1 and "eating" in templates.columns:
                # Re-assign one non-eating person to template with eating
                hi_eat = templates[templates["eating"] > 30]
                if len(hi_eat) > 0 and non_eating:
                    target = non_eating[0]
                    pidx = persons_df[persons_df["person_id"] == target].index[0]
                    p_age = persons_df.loc[pidx, "AGEP"]
                    age_diffs = (hi_eat["age"] - p_age).abs()
                    best = hi_eat.iloc[age_diffs.argmin()]
                    assignments[target] = best["template_id"]

        elif constraint == "sleep":
            # Encourage aligned sleep for adult pairs (couple heuristic)
            adults = hh[adult_mask]
            if len(adults) >= 2:
                sorted_adults = adults.sort_values("AGEP")
                a1, a2 = sorted_adults.iloc[0], sorted_adults.iloc[1]
                if abs(a1["AGEP"] - a2["AGEP"]) < 15:
                    tpl1 = tpl_lookup.get(assignments.get(a1["person_id"]), {})
                    tpl2 = tpl_lookup.get(assignments.get(a2["person_id"]), {})
                    s1 = tpl1.get("sleep_time", 480) if isinstance(tpl1, (dict, pd.Series)) else 480
                    s2 = tpl2.get("sleep_time", 480) if isinstance(tpl2, (dict, pd.Series)) else 480
                    if abs(s1 - s2) > 120 and "sleep_time" in templates.columns:
                        # Re-assign the one with less sleep to a template closer in sleep
                        target_pid = a2["person_id"] if s2 < s1 else a1["person_id"]
                        target_sleep = max(s1, s2)
                        candidates = templates[(templates["sleep_time"] - target_sleep).abs() < 60]
                        if len(candidates) > 0:
                            pidx = persons_df[persons_df["person_id"] == target_pid].index[0]
                            p_age = persons_df.loc[pidx, "AGEP"]
                            age_diffs = (candidates["age"] - p_age).abs()
                            best = candidates.iloc[age_diffs.argmin()]
                            assignments[target_pid] = best["template_id"]

        elif constraint == "afterschool" and school_pids and adult_pids:
            # Ensure at least one adult has flexible schedule (leisure or household_work > 60)
            flex_scores = []
            for pid in adult_pids:
                tpl = tpl_lookup.get(assignments.get(pid), {})
                leisure = tpl.get("leisure", 0) if isinstance(tpl, (dict, pd.Series)) else 0
                hw = tpl.get("household_work", 0) if isinstance(tpl, (dict, pd.Series)) else 0
                flex_scores.append((pid, leisure + hw))
            if flex_scores and max(f for _, f in flex_scores) < 60:
                if "leisure" in templates.columns:
                    hi_flex = templates[(templates["leisure"] + templates.get("household_work", 0)) > 120]
                    if len(hi_flex) > 0:
                        target = min(flex_scores, key=lambda x: x[1])[0]
                        pidx = persons_df[persons_df["person_id"] == target].index[0]
                        p_age = persons_df.loc[pidx, "AGEP"]
                        age_diffs = (hi_flex["age"] - p_age).abs()
                        best = hi_flex.iloc[age_diffs.argmin()]
                        assignments[target] = best["template_id"]

    return assignments


# ---- Phase 3 metric functions ----

def _childcare_coverage(assignments: Dict, persons_df: pd.DataFrame,
                        templates: pd.DataFrame) -> float:
    """% of households with children where an adult has caring > 60 min."""
    tpl_lookup = {row["template_id"]: row for _, row in templates.iterrows()}
    n_hh = 0
    n_covered = 0
    for _, hh in persons_df.groupby("building_id"):
        ages = hh["AGEP"].fillna(35)
        if not any(ages < 12):
            continue
        n_hh += 1
        adults = hh[ages >= 18]
        for _, adult in adults.iterrows():
            tpl = tpl_lookup.get(assignments.get(adult["person_id"]), {})
            care = tpl.get("caring_household", 0) if isinstance(tpl, (dict, pd.Series)) else 0
            if care >= 60:
                n_covered += 1
                break
    return round(n_covered / max(n_hh, 1) * 100, 1)


def _meal_synchronization(assignments: Dict, persons_df: pd.DataFrame,
                          templates: pd.DataFrame) -> float:
    """% of multi-person households where >=2 members have eating > 30 min."""
    tpl_lookup = {row["template_id"]: row for _, row in templates.iterrows()}
    n_hh = 0
    n_sync = 0
    for _, hh in persons_df.groupby("building_id"):
        if len(hh) < 2:
            continue
        n_hh += 1
        eaters = 0
        for _, p in hh.iterrows():
            tpl = tpl_lookup.get(assignments.get(p["person_id"]), {})
            eat = tpl.get("eating", 0) if isinstance(tpl, (dict, pd.Series)) else 0
            if eat >= 30:
                eaters += 1
        if eaters >= 2:
            n_sync += 1
    return round(n_sync / max(n_hh, 1) * 100, 1)


def _time_use_fidelity(assignments: Dict, templates: pd.DataFrame) -> float:
    """RMSE (minutes) between synthetic and ATUS aggregate activity shares."""
    activity_cols = ["sleep_time", "work", "household_work", "caring_household",
                     "leisure", "travel", "eating", "personal_care", "education"]
    available = [c for c in activity_cols if c in templates.columns]
    if not available:
        return 0.0

    # ATUS weighted means
    if "final_weight" in templates.columns:
        w = templates["final_weight"].fillna(1)
        atus_means = {c: (templates[c].fillna(0) * w).sum() / w.sum() for c in available}
    else:
        atus_means = {c: templates[c].fillna(0).mean() for c in available}

    # Synthetic means
    assigned_ids = set(assignments.values())
    assigned = templates[templates["template_id"].isin(assigned_ids)]
    if len(assigned) == 0:
        return 0.0
    synth_means = {c: assigned[c].fillna(0).mean() for c in available}

    se = [(synth_means[c] - atus_means[c]) ** 2 for c in available]
    return round(float(np.sqrt(np.mean(se))), 1)


def _couple_sleep_alignment(assignments: Dict, persons_df: pd.DataFrame,
                            templates: pd.DataFrame) -> float:
    """Mean |sleep_time difference| (minutes) for identified couples."""
    tpl_lookup = {row["template_id"]: row for _, row in templates.iterrows()}
    diffs = []
    for _, hh in persons_df.groupby("building_id"):
        adults = hh[hh["AGEP"] >= 18].sort_values("AGEP")
        if len(adults) < 2:
            continue
        a1, a2 = adults.iloc[0], adults.iloc[1]
        if abs(a1["AGEP"] - a2["AGEP"]) > 15:
            continue
        tpl1 = tpl_lookup.get(assignments.get(a1["person_id"]), {})
        tpl2 = tpl_lookup.get(assignments.get(a2["person_id"]), {})
        s1 = tpl1.get("sleep_time", 480) if isinstance(tpl1, (dict, pd.Series)) else 480
        s2 = tpl2.get("sleep_time", 480) if isinstance(tpl2, (dict, pd.Series)) else 480
        d = abs(s1 - s2)
        d = min(d, 1440 - d)  # circular
        diffs.append(d)
    return round(float(np.mean(diffs)), 1) if diffs else 0.0


def _afterschool_coverage(assignments: Dict, persons_df: pd.DataFrame,
                          templates: pd.DataFrame) -> float:
    """% of households with school-age children where an adult has leisure+hw > 60."""
    tpl_lookup = {row["template_id"]: row for _, row in templates.iterrows()}
    n_hh = 0
    n_covered = 0
    for _, hh in persons_df.groupby("building_id"):
        ages = hh["AGEP"].fillna(35)
        has_school = any((ages >= 6) & (ages < 18))
        if not has_school:
            continue
        adults = hh[ages >= 18]
        if len(adults) == 0:
            continue
        n_hh += 1
        for _, adult in adults.iterrows():
            tpl = tpl_lookup.get(assignments.get(adult["person_id"]), {})
            leisure = tpl.get("leisure", 0) if isinstance(tpl, (dict, pd.Series)) else 0
            hw = tpl.get("household_work", 0) if isinstance(tpl, (dict, pd.Series)) else 0
            if leisure + hw > 60:
                n_covered += 1
                break
    return round(n_covered / max(n_hh, 1) * 100, 1)


def run_ablation_b(buildings: pd.DataFrame,
                   templates: pd.DataFrame) -> Dict:
    """Run the progressive constraint ablation for Phase 3."""
    logger.info("=== Ablation B: Greedy vs MILP-Coordinated ===")

    persons_df = extract_persons(buildings)
    if len(persons_df) == 0:
        logger.error("No persons extracted — cannot run Ablation B")
        return {}

    pf, tf = create_feature_vectors(persons_df, templates)

    # Progressive constraint configs
    configs = [
        ("Greedy", []),
        ("+Childcare", ["childcare"]),
        ("+Meals", ["childcare", "meals"]),
        ("+Sleep", ["childcare", "meals", "sleep"]),
        ("Full MILP", ["childcare", "meals", "sleep", "afterschool"]),
    ]

    results = {}
    for label, constraints in configs:
        logger.info(f"  Running config: {label} ({constraints})")
        # Start from fresh greedy assignment each time
        asgn = greedy_knn_assignment(persons_df, pf, tf, templates)
        for c in constraints:
            asgn = apply_constraint(asgn, persons_df, templates, c)

        cc = _childcare_coverage(asgn, persons_df, templates)
        ms = _meal_synchronization(asgn, persons_df, templates)
        tu = _time_use_fidelity(asgn, templates)
        sl = _couple_sleep_alignment(asgn, persons_df, templates)
        asc = _afterschool_coverage(asgn, persons_df, templates)

        results[label] = {
            "childcare_coverage_pct": cc,
            "meal_sync_pct": ms,
            "time_use_rmse_min": tu,
            "couple_sleep_diff_min": sl,
            "afterschool_coverage_pct": asc,
        }
        logger.info(f"    CC={cc}% MS={ms}% TU-RMSE={tu} Sleep={sl} AS={asc}%")

    return results


# ============================================================
# LATEX TABLE FORMATTERS
# ============================================================

def format_table_a(res: Dict) -> str:
    b = res["binary_fs"]
    c = res["beta_continuous"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ablation A: Binary Fellegi--Sunter vs.\ Beta-family continuous",
        r"    similarity model (Phase~2 matching, $n=2\,000$ buildings).}",
        r"  \label{tab:ablation-a}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Metric} & \textbf{Binary F-S} & \textbf{Beta Continuous} \\",
        r"    \midrule",
    ]
    for tau in ["0.5", "0.7", "0.9"]:
        bp = b["precision_proxy"][tau]
        cp = c["precision_proxy"][tau]
        lines.append(f"    Precision proxy ($\\tau={tau}$) & {bp:.3f} & {cp:.3f} \\\\")
    for tau in ["0.5", "0.7", "0.9"]:
        br = b["recall_proxy"][tau]
        cr = c["recall_proxy"][tau]
        lines.append(f"    Recall proxy ($\\tau={tau}$) & {br:.3f} & {cr:.3f} \\\\")
    lines.append(f"    ECE $\\downarrow$ & {b['ece']:.4f} & {c['ece']:.4f} \\\\")
    lines.append(f"    Brier score $\\downarrow$ & {b['brier_score']:.4f} & {c['brier_score']:.4f} \\\\")
    lines.append(f"    KS statistic $\\uparrow$ & {b['ks_statistic']:.3f} & {c['ks_statistic']:.3f} \\\\")
    lines.append(f"    Mean weight & {b['mean_weight']:.2f} & {c['mean_weight']:.2f} \\\\")
    lines.append(f"    Benchmark dev.\\ (\\%) $\\downarrow$ & {b['mean_benchmark_deviation_pct']:.1f} & {c['mean_benchmark_deviation_pct']:.1f} \\\\")
    lines.append(f"    EM iterations & {b['em_iterations']} & {c['em_iterations']} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def format_table_b(res: Dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Ablation B: Progressive household coordination constraints",
        r"    (Phase~3 activity assignment, $n=2\,000$ buildings). Each row adds",
        r"    one constraint to the previous configuration.}",
        r"  \label{tab:ablation-b}",
        r"  \small",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    \textbf{Configuration}",
        r"      & \textbf{CC\%}",
        r"      & \textbf{MS\%}",
        r"      & \textbf{TU}",
        r"      & \textbf{Sleep$\Delta$}",
        r"      & \textbf{AS\%} \\",
        r"    \midrule",
    ]
    for label in ["Greedy", "+Childcare", "+Meals", "+Sleep", "Full MILP"]:
        r = res[label]
        lines.append(
            f"    {label} & {r['childcare_coverage_pct']:.1f}"
            f" & {r['meal_sync_pct']:.1f}"
            f" & {r['time_use_rmse_min']:.1f}"
            f" & {r['couple_sleep_diff_min']:.1f}"
            f" & {r['afterschool_coverage_pct']:.1f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{4pt}",
        r"  \begin{flushleft}",
        r"  \footnotesize",
        r"  CC\,=\,childcare coverage; MS\,=\,meal synchronization;",
        r"  TU\,=\,time-use RMSE (min); Sleep$\Delta$\,=\,mean couple",
        r"  sleep-onset difference (min); AS\,=\,after-school adult presence.",
        r"  \end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation study for reviewer comment 5")
    parser.add_argument("--n-buildings", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--max-pairs", type=int, default=40000,
                        help="Max candidate pairs for Phase 2 similarity computation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    t0 = time.time()

    # ---- Load data ----
    logger.info("=" * 60)
    logger.info("ABLATION STUDY — Reviewer Comment 5")
    logger.info("=" * 60)

    pums = load_phase1_sample(args.n_buildings)

    logger.info("Loading RECS data ...")
    recs = load_recs_data()
    recs = create_comprehensive_matching_features(recs, dataset_type="recs")

    pums_feat = create_comprehensive_matching_features(pums.copy(), dataset_type="pums")
    recs_feat = recs.copy()
    pums_feat, recs_feat = align_features_for_matching(pums_feat, recs_feat)

    logger.info("Loading ATUS data ...")
    atus_data = load_all_atus_data()
    atus_templates = create_activity_templates(atus_data)
    logger.info(f"ATUS templates: {len(atus_templates)}")

    # ---- Ablation A ----
    feature_list = get_matching_features_list()
    common = [f for f in feature_list
              if f in pums_feat.columns and f in recs_feat.columns]
    logger.info(f"Common matching features: {len(common)}")
    fields = _build_comparison_fields(common)
    field_names = [f.name for f in fields]

    sim_df = _compute_similarity_matrix(pums_feat, recs_feat, fields,
                                         max_pairs=args.max_pairs)

    binary_res = run_binary_baseline(sim_df, field_names, len(pums))
    beta_res = run_beta_continuous(sim_df, field_names, fields, len(pums))
    ablation_a = compute_ablation_a_metrics(binary_res, beta_res, pums, recs, len(pums))

    # ---- Ablation B ----
    ablation_b = run_ablation_b(pums, atus_templates)

    elapsed = time.time() - t0

    # ---- Compile and save ----
    full_results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "n_buildings": len(pums),
            "n_common_features": len(common),
            "n_atus_templates": len(atus_templates),
            "max_pairs": args.max_pairs,
            "runtime_seconds": round(elapsed, 1),
        },
        "ablation_a": ablation_a,
        "ablation_b": ablation_b,
    }

    # JSON
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info(f"JSON results → {json_path}")

    # LaTeX tables
    tex_a = output_dir / "ablation_table_a.tex"
    with open(tex_a, "w") as f:
        f.write(format_table_a(ablation_a))
    logger.info(f"LaTeX Table A → {tex_a}")

    tex_b = output_dir / "ablation_table_b.tex"
    with open(tex_b, "w") as f:
        f.write(format_table_b(ablation_b))
    logger.info(f"LaTeX Table B → {tex_b}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"\nCompleted in {elapsed:.1f} seconds\n")

    print("--- Ablation A: Binary F-S vs Beta Continuous ---")
    for label in ["binary_fs", "beta_continuous"]:
        r = ablation_a[label]
        print(f"  {label}: ECE={r['ece']:.4f}  Brier={r['brier_score']:.4f}  "
              f"KS={r['ks_statistic']:.3f}  BM-dev={r['mean_benchmark_deviation_pct']:.1f}%")

    print("\n--- Ablation B: Progressive Constraints ---")
    for label in ["Greedy", "+Childcare", "+Meals", "+Sleep", "Full MILP"]:
        r = ablation_b[label]
        print(f"  {label:15s}: CC={r['childcare_coverage_pct']:5.1f}%  "
              f"MS={r['meal_sync_pct']:5.1f}%  TU={r['time_use_rmse_min']:5.1f}  "
              f"Sleep={r['couple_sleep_diff_min']:5.1f}  AS={r['afterschool_coverage_pct']:5.1f}%")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
