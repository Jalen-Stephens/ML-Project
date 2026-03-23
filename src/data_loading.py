"""Download and parse the four SNAP BioSNAP TSV datasets."""

import os
import gzip
import requests
import pandas as pd
from src.utils import DATA_RAW

DATASETS = {
    "ChCh-Miner": {
        "url": "https://snap.stanford.edu/biodata/datasets/10001/files/ChCh-Miner_durgbank-chem-chem.tsv.gz",
        "filename": "ChCh-Miner_durgbank-chem-chem.tsv.gz",
        "description": "Drug-drug interactions",
    },
    "ChG-Miner": {
        "url": "https://snap.stanford.edu/biodata/datasets/10002/files/ChG-Miner_miner-chem-gene.tsv.gz",
        "filename": "ChG-Miner_miner-chem-gene.tsv.gz",
        "description": "Drug-gene (target) interactions",
    },
    "DG-AssocMiner": {
        "url": "https://snap.stanford.edu/biodata/datasets/10012/files/DG-AssocMiner_miner-disease-gene.tsv.gz",
        "filename": "DG-AssocMiner_miner-disease-gene.tsv.gz",
        "description": "Disease-gene associations",
    },
    "DCh-Miner": {
        "url": "https://snap.stanford.edu/biodata/datasets/10004/files/DCh-Miner_miner-disease-chemical.tsv.gz",
        "filename": "DCh-Miner_miner-disease-chemical.tsv.gz",
        "description": "Disease-drug associations",
    },
}


def download_dataset(name: str, force: bool = False) -> str:
    """Download a single dataset. Returns path to the .tsv.gz file."""
    info = DATASETS[name]
    os.makedirs(DATA_RAW, exist_ok=True)
    filepath = os.path.join(DATA_RAW, info["filename"])
    if os.path.exists(filepath) and not force:
        print(f"[skip] {name} already downloaded: {filepath}")
        return filepath
    print(f"[download] {name} from {info['url']} ...")
    resp = requests.get(info["url"], stream=True, timeout=120)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"[done] saved to {filepath}")
    return filepath


def download_all(force: bool = False) -> dict[str, str]:
    """Download all four datasets. Returns {name: filepath}."""
    return {name: download_dataset(name, force=force) for name in DATASETS}


def load_tsv(name: str) -> pd.DataFrame:
    """Load a downloaded dataset as a DataFrame with standardized columns."""
    info = DATASETS[name]
    filepath = os.path.join(DATA_RAW, info["filename"])
    if not os.path.exists(filepath):
        filepath = download_dataset(name)

    if name == "ChCh-Miner":
        # No header, two DrugBank ID columns
        df = pd.read_csv(filepath, sep="\t", compression="gzip",
                         header=None, names=["drug1", "drug2"])

    elif name == "ChG-Miner":
        # Header: #Drug  Gene (DrugBank ID, UniProt ID)
        df = pd.read_csv(filepath, sep="\t", compression="gzip",
                         comment="#", header=None, names=["drug", "gene"])

    elif name == "DG-AssocMiner":
        # Header: # Disease ID  "Disease Name"  Gene ID
        # 3 columns: CUI disease ID, disease name string, Entrez gene ID
        df = pd.read_csv(filepath, sep="\t", compression="gzip",
                         comment="#", header=None,
                         names=["disease", "disease_name", "gene"])
        df["gene"] = df["gene"].astype(str).str.strip()
        df["disease_name"] = df["disease_name"].str.strip('" ')
        # Prefix gene IDs to distinguish from UniProt IDs in ChG-Miner
        df["gene"] = "ENTREZ:" + df["gene"]

    elif name == "DCh-Miner":
        # Header: # Disease(MESH)  Chemical (MeSH disease ID, DrugBank ID)
        df = pd.read_csv(filepath, sep="\t", compression="gzip",
                         comment="#", header=None, names=["disease", "drug"])

    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = df.dropna().drop_duplicates()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    print(f"[loaded] {name}: {len(df)} edges, columns={list(df.columns)}")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load all four datasets. Returns {name: DataFrame}."""
    return {name: load_tsv(name) for name in DATASETS}
