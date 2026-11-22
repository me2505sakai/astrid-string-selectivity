import pandas as pd


def preprocess_no_preprcs(df: pd.DataFrame) -> pd.DataFrame:
    """前処理なし（そのまま返す）"""
    return df.reset_index(drop=True)


def preprocess_2024(df: pd.DataFrame) -> pd.DataFrame:
    """2024年手法: 文字列長3以上 かつ selectivity 100以上の行を100回複製（計101倍）"""
    strings = df["string"].astype(str)
    condition = (strings.str.len() >= 3) & (df["selectivity"] >= 100)
    matched_rows = df[condition]
    return pd.concat([df] + [matched_rows] * 100, ignore_index=True)


def preprocess_2025(df: pd.DataFrame) -> pd.DataFrame:
    """2025年手法: 文字列長 × selectivityの桁数 だけ各行を複製"""
    strings = df["string"].astype(str)
    string_lengths = strings.str.len()
    selectivity_digits = df["selectivity"].astype(str).str.len()
    replication_counts = string_lengths * selectivity_digits
    return df.loc[df.index.repeat(replication_counts)].reset_index(drop=True)


# 新しい前処理を追加する場合はここに関数を登録する
PREPROCESS_FUNCTIONS = {
    "no-preprcs": preprocess_no_preprcs,
    "2024preprcs": preprocess_2024,
    "2025preprcs": preprocess_2025,
}


def apply_preprocess(df: pd.DataFrame, preprocess_desc: str) -> pd.DataFrame:
    """指定された preprocess_desc に対応する前処理を DataFrame に適用する"""
    if preprocess_desc not in PREPROCESS_FUNCTIONS:
        available = list(PREPROCESS_FUNCTIONS.keys())
        raise ValueError(f"Unknown preprocess_desc: '{preprocess_desc}'. Available: {available}")

    return PREPROCESS_FUNCTIONS[preprocess_desc](df)