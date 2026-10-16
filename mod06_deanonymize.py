import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """

    excluded_columns = {"anon_id", "matched_name", "name"}

    quasi_identifiers = [col for col in anon_df.columns
                         if col in aux_df.columns and col not in excluded_columns]

    if not quasi_identifiers:
        return pd.DataFrame(columns=["anon_id", "matched_name"])

    merged = anon_df.merge(aux_df, on=quasi_identifiers, how="inner", suffixes=("_anon", "_aux"))

    match_counts = merged.groupby("anon_id")["name"].transform("size")
    unique_matches = merged[match_counts == 1]

    matches_df = unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})
    matches_df = matches_df.drop_duplicates(subset=["anon_id"])

    return matches_df.reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    total_records = len(anon_df)
    if total_records == 0: return 0.0
    uniquely_matched = matches_df["anon_id"].nunique()
    return uniquely_matched / total_records
