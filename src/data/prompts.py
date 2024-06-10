from langchain.prompts import PromptTemplate
import pandas as pd


# TEMPLATE = """{speaker_name} ({speaker_affiliation}{speaker_job}{speaker_state}) said the statement: "{statement}"{statement_context}"""
TEMPLATE = (
    """{speaker_name} ({speaker_affiliation}{speaker_job}{speaker_state}) said the statement: "{statement}"{statement_context}{subjects}"""
)
# TEMPLATE = 'a speaker ({speaker_affiliation}{speaker_job}) said the statement: "{statement}"'
# TEMPLATE = """{speaker_name} ({speaker_affiliation}) said the statement: "{statement}"{statement_context}"""
# TEMPLATE = 'a speaker ({speaker_affiliation}) said the statement: "{statement}"'
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
    speaker_name = format_speaker_name(row["speaker_name"])
    speaker_affiliation = format_speaker_affiliation(row["speaker_affiliation"])
    speaker_job = format_speaker_job(row["speaker_job"])
    speaker_state = format_speaker_state(row["speaker_state"])
    statement_context = format_statement_context(row["statement_context"])
    subjects = format_subjects(row["subjects"])

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


def format_speaker_name(speaker_name):
    return f"""the speaker {speaker_name.replace("-", " ")}""" if pd.notna(speaker_name) else "a speaker"


def format_speaker_affiliation(speaker_affiliation):
    return f"{speaker_affiliation.replace('-', ' ')} " if pd.notna(speaker_affiliation) else ""


def format_speaker_job(speaker_job):
    return f"{speaker_job} " if pd.notna(speaker_job) else ""


def format_speaker_state(speaker_state):
    return f"from {speaker_state} state" if pd.notna(speaker_state) else ""


def format_statement_context(statement_context):
    return f" in the context of {statement_context}" if pd.notna(statement_context) else ""


def format_subjects(subjects):
    formatted_subjects = ", ".join(s.replace("-", " ") for s in subjects.split("$")) if pd.notna(subjects) else ""
    return f" while talking about {formatted_subjects}" if formatted_subjects else ""
