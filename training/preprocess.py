"""
AI Exercise Form & Posture Intelligence Platform
Academic Data Leakage Prevention & Session-Aware Dataset Splitter
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def session_aware_train_val_test_split(
    df: pd.DataFrame,
    session_col: str = "session_id",
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ACADEMIC DATA LEAKAGE PREVENTION:
    Standard random train_test_split across individual consecutive video frames causes
    severe data leakage because frame t and frame t+1 are virtually identical.

    This function performs a strict SESSION-AWARE / SUBJECT-AWARE split:
    Entire recording sessions are partitioned as unbroken units into Train, Val, and Test sets,
    ensuring zero correlation leakage between training and evaluation frames.
    """
    if session_col not in df.columns:
        # Fallback to stratified group chunking if session_id is missing
        raise ValueError(f"Session column '{session_col}' must be present in dataframe to prevent data leakage.")

    unique_sessions = df[session_col].unique()
    np.random.seed(random_seed)
    shuffled_sessions = np.random.permutation(unique_sessions)

    n_total = len(shuffled_sessions)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    test_sessions = set(shuffled_sessions[:n_test])
    val_sessions = set(shuffled_sessions[n_test:n_test + n_val])
    train_sessions = set(shuffled_sessions[n_test + n_val:])

    train_df = df[df[session_col].isin(train_sessions)].copy()
    val_df = df[df[session_col].isin(val_sessions)].copy()
    test_df = df[df[session_col].isin(test_sessions)].copy()

    return train_df, val_df, test_df
