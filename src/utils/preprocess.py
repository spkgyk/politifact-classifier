from joblib import Parallel, delayed
import pandas as pd

STATE_MAPPING = {
    "tex": "texas",
    "the united states": "united states",
    "washington state": "washington",
    "washington d.c.": "district of columbia",
    "washington dc": "district of columbia",
    "washington, d.c.": "district of columbia",
    "virgina": "virginia",
    "virginia director, coalition to stop gun violence": "virginia",
}


def standardize_state(state):
    if pd.isnull(state):
        return state
    return STATE_MAPPING.get(state, state)


def process_subjects(df):
    unique_subjects = set()
    for subjects in df["subjects"]:
        for subject in subjects.split("$"):
            unique_subjects.add("subject-" + subject)

    subjects_data = {}
    for subject in sorted(list(unique_subjects)):
        subjects_data[subject] = df["subjects"].apply(lambda x: int(subject in x))

    subjects_df = pd.DataFrame(subjects_data)

    # Step 3: Concatenate the new columns with the original DataFrame
    df = pd.concat([df, subjects_df], axis="columns")

    return df


def preprocess_data(df: pd.DataFrame):
    df = df.map(lambda x: x.lower().replace('"', "").strip() if isinstance(x, str) else x)
    df["label"] = df["Label"].apply(lambda x: int(x.lower() in ["true", "mostly-true"]))
    df["speaker_state"] = df["speaker_state"].map(standardize_state)
    df = process_subjects(df)
    df["statement_context"] = df["statement_context"].fillna("")

    if "statement_embedding" in df.columns:
        df["statement_embedding"] = Parallel(n_jobs=-1)(delayed(eval)(row["statement_embedding"]) for _, row in df.iterrows())

    df = pd.DataFrame(df)

    return df
