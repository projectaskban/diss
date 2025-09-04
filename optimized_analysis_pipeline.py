#!/usr/bin/env python3
"""
Single-Cell RNA-seq Aging Analysis Pipeline
Analyzes transcriptional burst parameters across young/old mice
using dual-platform (10x and Smart-seq2) validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
import gseapy as gp
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration for all analysis parameters"""
    # Paths
    ROOT = Path("/Users/david-johnlonge/Documents/Uni stuff/diss/8273102")
    OUTDIR = ROOT / "results"
    TABDIR = OUTDIR / "tables"
    FIGDIR = OUTDIR / "figures"
    LISTDIR = TABDIR / "gene_lists"
    ENRDIR = TABDIR / "enrichment"
    
    # Create directories
    for dir_path in [TABDIR, FIGDIR, LISTDIR, ENRDIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Analysis parameters
    ABS_MIN = 0.3  # minimum absolute log2FC for concordance
    FDR_CUTOFF = 0.05
    BOOTSTRAP_ITERATIONS = 300
    TRIM_QUANTILE = 0.99  # for visualization
    
    # Enrichment libraries
    ENRICHR_LIBS = [
        "MSigDB_Hallmark_2020",
        "GO_Biological_Process_2021", 
        "KEGG_2021_Mouse",
        "WikiPathways_2023_Mouse",
        "Reactome_2022"
    ]
    
    # Random seeds for reproducibility
    SEED_10X = 123
    SEED_SS2 = 456

# ============================================================================
# Data Processing Functions
# ============================================================================

def load_kinetics_data(tissue: str, cell_supertype: str) -> pd.DataFrame:
    """Load pre-computed kinetics data for a specific tissue/cell type"""
    filepath = Config.TABDIR / f"kinetics_{tissue}_{cell_supertype}.csv"
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    return pd.read_csv(filepath)

def strict_mask(df: pd.DataFrame, param: str, cutoff: float) -> pd.Series:
    """Create boolean mask for significant genes with strict thresholds"""
    return (
        (df[f"log2FC_{param}"].abs() >= cutoff) & 
        (df[f"fdr_{param}"] < Config.FDR_CUTOFF)
    )

def downsample_counts(X, target: int, seed: int = None):
    """
    Downsample count matrix to target depth using binomial sampling.
    Works with both sparse and dense matrices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        data = X.data
        indptr = X.indptr
        
        for i in range(X.shape[0]):
            s, e = indptr[i], indptr[i+1]
            if e <= s:  # empty row
                continue
            row_sum = data[s:e].sum()
            if row_sum > target and row_sum > 0:
                p = float(target) / float(row_sum)
                vals = data[s:e].astype(np.int64)
                data[s:e] = np.random.binomial(vals, p).astype(vals.dtype)
        
        X.eliminate_zeros()
        return X
    else:
        X = np.array(X, copy=True)
        if not np.issubdtype(X.dtype, np.integer):
            X = np.rint(X).astype(np.int64)
        
        for i in range(X.shape[0]):
            tot = X[i].sum()
            if tot > target and tot > 0:
                p = float(target) / float(tot)
                X[i] = np.random.binomial(X[i].astype(np.int64), p)
        return X

# ============================================================================
# Cross-Platform Concordance Analysis
# ============================================================================

def compute_platform_concordance():
    """Analyze cross-platform reproducibility between 10x and Smart-seq2"""
    
    # Load optimal cutoffs
    cutoffs_df = pd.read_csv(Config.TABDIR / "optimal_cutoffs.csv")
    cutoffs = {(r["tissue"], r["cell_supertype"], r["param"]): r["cutoff"] 
               for _, r in cutoffs_df.iterrows()}
    
    concordance_results = []
    
    # Get all available tissue/cell type combinations
    kinetics_files = list(Config.TABDIR.glob("kinetics_*.csv"))
    
    for kinetics_file in kinetics_files:
        # Parse tissue and cell type from filename
        parts = kinetics_file.stem.replace("kinetics_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        tissue, cell_type = parts
        
        # Load data for both platforms
        df = pd.read_csv(kinetics_file)
        if df.empty:
            continue
            
        d10x = df[df.platform == "10x"].copy()
        dss2 = df[df.platform == "Smart-seq2"].copy()
        
        if d10x.empty or dss2.empty:
            continue
        
        # Get cutoffs for this stratum
        ca = cutoffs.get((tissue, cell_type, "a"), Config.ABS_MIN)
        cb = cutoffs.get((tissue, cell_type, "b"), Config.ABS_MIN)
        
        # Apply significance filters
        m10a = strict_mask(d10x, "a", Config.ABS_MIN)
        m10b = strict_mask(d10x, "b", Config.ABS_MIN)
        mSSa = strict_mask(dss2, "a", ca)
        mSSb = strict_mask(dss2, "b", cb)
        
        # Merge significant genes from both platforms
        A = d10x.loc[m10a, ["gene", "log2FC_a"]].merge(
            dss2.loc[mSSa, ["gene", "log2FC_a"]], 
            on="gene", suffixes=("_10x", "_SS2")
        )
        B = d10x.loc[m10b, ["gene", "log2FC_b"]].merge(
            dss2.loc[mSSb, ["gene", "log2FC_b"]], 
            on="gene", suffixes=("_10x", "_SS2")
        )
        
        # Find concordant genes (same direction of change)
        A_conc = A[np.sign(A["log2FC_a_10x"]) == np.sign(A["log2FC_a_SS2"])].copy()
        B_conc = B[np.sign(B["log2FC_b_10x"]) == np.sign(B["log2FC_b_SS2"])].copy()
        
        # Save concordant gene lists
        if len(A_conc) > 0:
            A_conc.assign(
                mean_abs=lambda x: (x["log2FC_a_10x"].abs() + x["log2FC_a_SS2"].abs()) / 2
            ).sort_values("mean_abs", ascending=False)["gene"].to_csv(
                Config.LISTDIR / f"{tissue}__{cell_type}__concordant_a_genes.txt",
                index=False, header=False
            )
        
        if len(B_conc) > 0:
            B_conc.assign(
                mean_abs=lambda x: (x["log2FC_b_10x"].abs() + x["log2FC_b_SS2"].abs()) / 2
            ).sort_values("mean_abs", ascending=False)["gene"].to_csv(
                Config.LISTDIR / f"{tissue}__{cell_type}__concordant_b_genes.txt",
                index=False, header=False
            )
        
        concordance_results.append({
            "tissue": tissue,
            "cell_supertype": cell_type,
            "n_overlap_a": len(A),
            "n_concordant_a": len(A_conc),
            "n_overlap_b": len(B),
            "n_concordant_b": len(B_conc)
        })
    
    # Save concordance summary
    if concordance_results:
        concord_df = pd.DataFrame(concordance_results).sort_values(
            ["n_concordant_a", "n_concordant_b"], ascending=False
        ).reset_index(drop=True)
        
        concord_df.to_csv(Config.TABDIR / "platform_concordant_hits.csv", index=False)
        print(f"Saved concordance analysis: {Config.TABDIR / 'platform_concordant_hits.csv'}")
        return concord_df
    
    return pd.DataFrame()

# ============================================================================
# Pathway Enrichment Analysis
# ============================================================================

def tidy_enrichr_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Enrichr output"""
    keep_cols = ["Term", "Adjusted P-value", "P-value", "Combined Score", 
                 "Overlap", "Odds Ratio", "Genes"]
    
    rename_map = {
        "Term": "term",
        "Adjusted P-value": "fdr",
        "P-value": "pval",
        "Combined Score": "combined",
        "Overlap": "overlap",
        "Odds Ratio": "odds",
        "Genes": "genes"
    }
    
    return df[keep_cols].rename(columns=rename_map).sort_values(
        ["fdr", "combined"]
    ).reset_index(drop=True)

def run_enrichment_analysis():
    """Perform pathway enrichment on concordant gene sets"""
    
    all_results = []
    gene_files = sorted(Config.LISTDIR.glob("*__concordant_*_genes.txt"))
    
    for gene_file in gene_files:
        # Parse metadata from filename
        parts = gene_file.stem.split("__")
        tissue = parts[0]
        supertype = parts[1]
        param = parts[2].replace("concordant_", "").replace("_genes", "")
        
        # Load gene list
        genes = [g.strip() for g in gene_file.read_text().splitlines() if g.strip()]
        
        # Skip small gene lists
        if len(genes) < 10:
            print(f"Skipping {gene_file.name}: only {len(genes)} genes")
            continue
        
        # Create output subdirectory
        subdir = Config.ENRDIR / f"{tissue}__{supertype}"
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Run enrichment for each library
        for lib in Config.ENRICHR_LIBS:
            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    description=f"{tissue}__{supertype}__{param}",
                    gene_sets=lib,
                    organism="Mouse",
                    outdir=None,
                    cutoff=1.0,
                    no_plot=True
                )
                
                if enr is None or enr.results is None or enr.results.empty:
                    continue
                
                # Process results
                res = tidy_enrichr_results(enr.results)
                res.insert(0, "library", lib)
                res.insert(0, "param", param)
                res.insert(0, "cell_supertype", supertype)
                res.insert(0, "tissue", tissue)
                
                # Save individual results
                res.to_csv(subdir / f"enrichr_{param}_{lib}.csv", index=False)
                all_results.append(res)
                
                # Create visualization for top terms
                create_enrichment_barplot(res, tissue, supertype, param, lib)
                
            except Exception as e:
                print(f"Error processing {lib} for {tissue}/{supertype}/{param}: {e}")
                continue
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(Config.ENRDIR / "enrichr_all_concordant.csv", index=False)
        print(f"Saved combined enrichment: {Config.ENRDIR / 'enrichr_all_concordant.csv'}")
        return combined
    
    return pd.DataFrame()

def create_enrichment_barplot(df: pd.DataFrame, tissue: str, supertype: str, 
                              param: str, lib: str, top_n: int = 10):
    """Create barplot for top enriched terms"""
    
    top = df.nsmallest(top_n, "fdr").copy()
    if top.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate -log10(FDR) with floor at 1e-300
    y_vals = -np.log10(top["fdr"].clip(lower=1e-300))
    
    # Create horizontal barplot
    bars = ax.barh(range(len(top)), y_vals, color='steelblue')
    
    # Customize appearance
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["term"], fontsize=10)
    ax.set_xlabel("-log10(FDR)", fontsize=12)
    ax.set_title(f"{tissue} • {supertype} • {param} • {lib}", fontsize=12)
    
    # Add significance line
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='FDR=0.05')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Config.FIGDIR / f"enrichr_{tissue}_{supertype}_{param}_{lib}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# Visualization Functions
# ============================================================================

def create_parameter_violin(tissue: str, supertype: str, param: str = "a"):
    """Create violin plot comparing parameter distributions between ages and platforms"""
    
    df = load_kinetics_data(tissue, supertype)
    if df.empty:
        print(f"No data available for {tissue}/{supertype}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    positions = []
    data_arrays = []
    labels = []
    colors = []
    pos = 0
    
    for plat in ["10x", "Smart-seq2"]:
        sub = df[df.platform == plat]
        if sub.empty:
            pos += 3
            continue
        
        # Extract age-specific values
        key_old = f"{param}_old"
        key_yng = f"{param}_yng"
        
        y_old = sub[key_old].dropna().values
        y_yng = sub[key_yng].dropna().values
        
        # Trim outliers for visualization
        if len(y_old) > 0:
            cutoff = np.quantile(y_old, Config.TRIM_QUANTILE)
            y_old = y_old[y_old <= cutoff]
        
        if len(y_yng) > 0:
            cutoff = np.quantile(y_yng, Config.TRIM_QUANTILE)
            y_yng = y_yng[y_yng <= cutoff]
        
        # Add to plot data
        if len(y_old) > 0:
            data_arrays.append(y_old)
            positions.append(pos)
            labels.append(f"{plat}\nOld")
            colors.append('darkred')
        
        if len(y_yng) > 0:
            data_arrays.append(y_yng)
            positions.append(pos + 1)
            labels.append(f"{plat}\nYoung")
            colors.append('darkblue')
        
        pos += 3
    
    if not data_arrays:
        print(f"No valid data for plotting {tissue}/{supertype}/{param}")
        return
    
    # Create violin plot
    parts = ax.violinplot(data_arrays, positions=positions, showextrema=False, widths=0.8)
    
    # Color the violins
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add boxplot overlay
    bp = ax.boxplot(data_arrays, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='white', alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))
    
    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Burst frequency (α̂)" if param == "a" else "Burst size (b̂)", fontsize=12)
    ax.set_title(f"{tissue} — {supertype}", fontsize=14)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Config.FIGDIR / f"violin_{param}_{tissue}_{supertype}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved violin plot: {fig_path}")

def create_replication_scatter(tissue: str, supertype: str):
    """Create scatter plot showing cross-platform replication"""
    
    df = load_kinetics_data(tissue, supertype)
    if df.empty:
        return
    
    # Prepare data for both platforms
    d10x = df[df.platform == "10x"][["gene", "log2FC_a", "log2FC_b"]].rename(
        columns={"log2FC_a": "a10", "log2FC_b": "b10"}
    )
    dss2 = df[df.platform == "Smart-seq2"][["gene", "log2FC_a", "log2FC_b"]].rename(
        columns={"log2FC_a": "aSS", "log2FC_b": "bSS"}
    )
    
    # Merge on common genes
    merged = d10x.merge(dss2, on="gene", how="inner").dropna()
    
    if len(merged) < 20:
        print(f"Insufficient overlap for {tissue}/{supertype}: {len(merged)} genes")
        return
    
    # Create figure with subplots for both parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot burst frequency (parameter a)
    ax1.scatter(merged["a10"], merged["aSS"], alpha=0.5, s=20, color='steelblue')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add correlation
    rho_a = merged[["a10", "aSS"]].corr(method="spearman").iloc[0, 1]
    ax1.text(0.05, 0.95, f"ρ = {rho_a:.3f}", transform=ax1.transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    ax1.set_xlabel("10x log2FC(α)", fontsize=11)
    ax1.set_ylabel("Smart-seq2 log2FC(α)", fontsize=11)
    ax1.set_title("Burst Frequency", fontsize=12)
    
    # Plot burst size (parameter b)
    ax2.scatter(merged["b10"], merged["bSS"], alpha=0.5, s=20, color='coral')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add correlation
    rho_b = merged[["b10", "bSS"]].corr(method="spearman").iloc[0, 1]
    ax2.text(0.05, 0.95, f"ρ = {rho_b:.3f}", transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    ax2.set_xlabel("10x log2FC(b)", fontsize=11)
    ax2.set_ylabel("Smart-seq2 log2FC(b)", fontsize=11)
    ax2.set_title("Burst Size", fontsize=12)
    
    # Overall title
    fig.suptitle(f"{tissue} — {supertype} (n={len(merged)} genes)", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Config.FIGDIR / f"replication_scatter_{tissue}_{supertype}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scatter plot: {fig_path}")

# ============================================================================
# Summary Statistics
# ============================================================================

def compute_summary_statistics():
    """Generate summary statistics for the analysis"""
    
    stats = {}
    
    # Load concordance data
    concord_path = Config.TABDIR / "platform_concordant_hits.csv"
    if concord_path.exists():
        concord = pd.read_csv(concord_path)
        stats["total_strata"] = len(concord)
        stats["median_concordant_a"] = concord["n_concordant_a"].median()
        stats["median_concordant_b"] = concord["n_concordant_b"].median()
        stats["total_concordant_a"] = concord["n_concordant_a"].sum()
        stats["total_concordant_b"] = concord["n_concordant_b"].sum()
    
    # Load enrichment data
    enrich_path = Config.ENRDIR / "enrichr_all_concordant.csv"
    if enrich_path.exists():
        enrich = pd.read_csv(enrich_path)
        sig_enrich = enrich[enrich["fdr"] < Config.FDR_CUTOFF]
        stats["total_enriched_terms"] = len(sig_enrich)
        stats["unique_pathways"] = sig_enrich["term"].nunique()
        
        # Top enriched libraries
        lib_counts = sig_enrich.groupby("library").size().sort_values(ascending=False)
        stats["top_library"] = lib_counts.index[0] if len(lib_counts) > 0 else "None"
        stats["top_library_count"] = lib_counts.iloc[0] if len(lib_counts) > 0 else 0
    
    # Create summary report
    summary = pd.Series(stats).to_frame("Value")
    summary.index.name = "Metric"
    summary.to_csv(Config.TABDIR / "analysis_summary.csv")
    
    print("\n=== Analysis Summary ===")
    print(summary)
    
    return summary

# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline():
    """Execute complete analysis pipeline"""
    
    print("=" * 60)
    print("Single-Cell Aging Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Cross-platform concordance
    print("\n[1/4] Computing cross-platform concordance...")
    concordance = compute_platform_concordance()
    if not concordance.empty:
        print(f"✓ Analyzed {len(concordance)} tissue/cell type combinations")
    
    # Step 2: Pathway enrichment
    print("\n[2/4] Running pathway enrichment analysis...")
    enrichment = run_enrichment_analysis()
    if not enrichment.empty:
        sig_terms = enrichment[enrichment["fdr"] < Config.FDR_CUTOFF]
        print(f"✓ Found {len(sig_terms)} significant enrichments")
    
    # Step 3: Generate visualizations
    print("\n[3/4] Creating visualizations...")
    
    # Get top strata for visualization
    if not concordance.empty:
        top_strata = concordance.nlargest(5, "n_concordant_a")
        for _, row in top_strata.iterrows():
            tissue = row["tissue"]
            cell_type = row["cell_supertype"]
            
            # Create violin plots
            create_parameter_violin(tissue, cell_type, "a")
            create_parameter_violin(tissue, cell_type, "b")
            
            # Create replication scatter
            create_replication_scatter(tissue, cell_type)
    
    # Step 4: Summary statistics
    print("\n[4/4] Computing summary statistics...")
    summary = compute_summary_statistics()
    
    print("\n" + "=" * 60)
    print("Pipeline complete! Results saved to:")
    print(f"  Tables: {Config.TABDIR}")
    print(f"  Figures: {Config.FIGDIR}")
    print(f"  Gene lists: {Config.LISTDIR}")
    print(f"  Enrichment: {Config.ENRDIR}")
    print("=" * 60)

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_full_pipeline()
