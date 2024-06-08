from langchain.prompts import PromptTemplate
import pandas as pd


# TEMPLATE = """{speaker_name} ({speaker_affiliation}{speaker_job}{speaker_state}) said the statement: "{statement}"{statement_context}"""
# TEMPLATE = (
#     """{speaker_name} ({speaker_affiliation}{speaker_job}{speaker_state}) said the statement: "{statement}"{statement_context}{subjects}"""
# )
TEMPLATE = 'A speaker ({speaker_affiliation}{speaker_job}) said the statement: "{statement}"'
# TEMPLATE = 'A speaker ({speaker_affiliation}{speaker_job}) said the statement: "{statement}". Is it true or false?'
PROMPT = PromptTemplate(
    input_variables=[
        "speaker_name",
        "speaker_affiliation",
        "speaker_job",
        "speaker_state",
        "statement",
        "statement_context",
        "subjects",
    ],
    template=TEMPLATE,
)


def create_prompt(row):
    speaker_name = f"""the speaker {row["speaker_name"].replace("-", " ")}""" if pd.notna(row["speaker_name"]) else "a speaker"
    speaker_affiliation = f"{row['speaker_affiliation'].replace('-', ' ')} " if pd.notna(row["speaker_affiliation"]) else ""
    speaker_job = f"{row['speaker_job']} " if pd.notna(row["speaker_job"]) else ""
    speaker_state = f"from {row['speaker_state']} state" if pd.notna(row["speaker_state"]) else ""
    statement_context = f" in the context of {row['statement_context']}" if pd.notna(row["statement_context"]) else ""
    subjects = ", ".join(s.replace("-", " ") for s in row["subjects"].split("$")) if pd.notna(row["subjects"]) else ""
    subjects = f" while talking about {subjects}" if subjects else subjects
    inputs = {
        "speaker_name": speaker_name,
        "speaker_affiliation": speaker_affiliation,
        "speaker_job": speaker_job,
        "speaker_state": speaker_state,
        "statement": row["statement"],
        "statement_context": statement_context,
        "subjects": subjects,
    }

    return PROMPT.format(**inputs).lower()
