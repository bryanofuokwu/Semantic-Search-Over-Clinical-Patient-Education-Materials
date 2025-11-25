import os
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple

import pandas as pd


# -----------------------------
# Configuration / constants
# -----------------------------

# Default number of documents to generate if not provided via CLI
DEFAULT_N_DOCS = 500

# Default output path (relative to project root)
DEFAULT_OUTPUT_PATH = os.path.join(
    "data",
    "raw",
    "patient_education.parquet",
)

# A small set of common chronic conditions and categories
CONDITIONS: List[Tuple[str, str]] = [
    ("Hypertension", "chronic_condition"),
    ("Type 2 Diabetes", "chronic_condition"),
    ("Asthma", "chronic_condition"),
    ("GERD (Acid Reflux)", "chronic_condition"),
    ("High Cholesterol", "chronic_condition"),
    ("Osteoarthritis", "chronic_condition"),
    ("Migraine", "chronic_condition"),
    ("Chronic Low Back Pain", "chronic_condition"),
    ("Generalized Anxiety", "mental_health"),
    ("Mild Depression", "mental_health"),
]

LIFESTYLE_THEMES = [
    "nutrition",
    "movement and exercise",
    "sleep habits",
    "stress management",
    "smoking cessation",
    "alcohol moderation",
]

READING_LEVELS = ["simple", "standard"]  # could be used later for personalization


# -----------------------------
# Data model
# -----------------------------


@dataclass
class PatientEducationDoc:
    """
    Represents a synthetic patient education leaflet for a single condition.
    """
    id: int
    condition: str
    category: str
    title: str
    reading_level: str
    overview: str
    symptoms: str
    causes: str
    diagnosis: str
    treatment_options: str
    self_care: str
    when_to_seek_help: str
    faq: str

    def to_full_text(self) -> str:
        """
        Optional helper if you later want to flatten sections into a single string.
        """
        parts = [
            f"Title: {self.title}",
            "",
            "Overview:",
            self.overview,
            "",
            "Symptoms:",
            self.symptoms,
            "",
            "Causes:",
            self.causes,
            "",
            "Diagnosis:",
            self.diagnosis,
            "",
            "Treatment Options:",
            self.treatment_options,
            "",
            "Self-Care and Daily Management:",
            self.self_care,
            "",
            "When to Seek Help:",
            self.when_to_seek_help,
            "",
            "Frequently Asked Questions:",
            self.faq,
        ]
        return "\n".join(parts)


# -----------------------------
# Section generators
# -----------------------------


def generate_overview(condition: str, level: str) -> str:
    if level == "simple":
        return (
            f"{condition} is a health problem that can affect how you feel and how your body works. "
            f"This guide explains {condition.lower()} in everyday language so you can better "
            f"understand what it is and what you can do about it."
        )
    else:
        return (
            f"{condition} is a common condition that develops over time and can affect your long-term health. "
            f"In this guide, we explain what {condition.lower()} is, why it matters, and how it can be "
            f"managed with a combination of medical care and daily habits."
        )


def generate_symptoms(condition: str, level: str) -> str:
    base = (
        f"People with {condition.lower()} may notice a range of symptoms, and some may have no clear "
        f"symptoms at all. Common signs can include changes in how you feel day to day and subtle "
        f"signals that something in your body is off. Because symptoms can be mild or easy to miss, "
        f"regular check-ins with a healthcare professional are important."
    )
    if level == "simple":
        return (
            base
            + " If you ever feel sudden, strong, or worrying symptoms, it is a good idea to seek medical help."
        )
    return base + " New or worsening symptoms should be discussed with your care team."


def generate_causes(condition: str) -> str:
    return (
        f"{condition} usually develops due to a mix of factors rather than a single cause. "
        f"These factors can include family history, other health conditions, long-term stress, "
        f"and daily habits around food, movement, sleep, and substances. In many cases, there is "
        f"no one person or event to blame—health is shaped over time."
    )


def generate_diagnosis(condition: str, level: str) -> str:
    if level == "simple":
        return (
            f"To check for {condition.lower()}, healthcare professionals often talk with you about your "
            f"symptoms, ask about your health history, and may do tests. These tests can include blood "
            f"work, scans, or other measurements. The goal is to understand the bigger picture, not just "
            f"one result."
        )
    return (
        f"Diagnosis of {condition.lower()} usually involves a combination of a detailed medical history, "
        f"physical examination, and targeted tests. These might include blood tests, imaging, or other "
        f"specific measurements. Results are interpreted in context, and your care team may repeat tests "
        f"over time to monitor changes."
    )


def generate_treatment_options(condition: str) -> str:
    lifestyle = random.choice(LIFESTYLE_THEMES)
    return (
        f"Treatment for {condition.lower()} often includes both daily lifestyle steps and, when appropriate, "
        f"medications. Lifestyle changes such as improving {lifestyle}, building regular activity into your "
        f"week, and creating routines that reduce stress can have a real impact. Medicines, when used, are "
        f"chosen to match your health history and are reviewed regularly with your healthcare team."
    )


def generate_self_care(condition: str, level: str) -> str:
    text = (
        f"Self-care for {condition.lower()} focuses on small, repeatable actions that support your health. "
        f"This may include following your care plan, taking medicines as directed, keeping track of symptoms, "
        f"and paying attention to how sleep, food, and movement affect how you feel. "
    )
    if level == "simple":
        text += (
            "It can help to write things down, set reminders, and ask friends or family to support you as "
            "you build new habits."
        )
    else:
        text += (
            "Many people find it useful to track patterns over time and to bring notes or questions to "
            "medical appointments."
        )
    return text


def generate_when_to_seek_help(condition: str) -> str:
    return (
        f"It is important to know when to contact a healthcare professional if you live with "
        f"{condition.lower()}. Sudden or severe changes in how you feel, new or worrying symptoms, "
        f"or any feeling that 'something is not right' are reasons to reach out. If you think you are "
        f"having a medical emergency, follow local emergency guidance right away rather than waiting."
    )


def generate_faq(condition: str, level: str) -> str:
    if level == "simple":
        q1 = (
            f"Q: Can {condition.lower()} be managed?\n"
            f"A: Many people live well with {condition.lower()} by working with their care team and making "
            f"small, realistic changes in daily life.\n"
        )
        q2 = (
            "Q: Will I need treatment forever?\n"
            "A: Treatment plans can change over time. Your healthcare professional can explain what to expect "
            "and help you review options as your needs change.\n"
        )
    else:
        q1 = (
            f"Q: Is {condition.lower()} a lifelong condition?\n"
            f"A: For many people, {condition.lower()} is long-term, but symptoms and risks can often be reduced "
            f"with consistent management and follow-up.\n"
        )
        q2 = (
            "Q: How often should I check in with my care team?\n"
            "A: The right follow-up schedule depends on your overall health, test results, and treatment plan. "
            "Your care team can recommend how often to be seen and when to schedule earlier visits."
        )

    q3 = (
        "\nQ: What if I feel overwhelmed or unsure?\n"
        "A: It is normal to have questions or mixed feelings about a diagnosis. Writing down concerns, "
        "bringing a trusted person to appointments, and asking your care team to explain things in a different "
        "way can all help."
    )

    return q1 + "\n" + q2 + q3


# -----------------------------
# Document generation
# -----------------------------


def generate_docs(n_docs: int, seed: int = 42) -> List[PatientEducationDoc]:
    """
    Generate a list of synthetic patient education documents.
    """
    random.seed(seed)
    docs: List[PatientEducationDoc] = []

    for i in range(n_docs):
        condition, category = random.choice(CONDITIONS)
        reading_level = random.choice(READING_LEVELS)
        title = f"Understanding {condition} – Patient Education Guide"

        doc = PatientEducationDoc(
            id=i,
            condition=condition,
            category=category,
            title=title,
            reading_level=reading_level,
            overview=generate_overview(condition, reading_level),
            symptoms=generate_symptoms(condition, reading_level),
            causes=generate_causes(condition),
            diagnosis=generate_diagnosis(condition, reading_level),
            treatment_options=generate_treatment_options(condition),
            self_care=generate_self_care(condition, reading_level),
            when_to_seek_help=generate_when_to_seek_help(condition),
            faq=generate_faq(condition, reading_level),
        )
        docs.append(doc)

    return docs


# -----------------------------
# CLI / entrypoint
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient education documents for semantic search."
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=DEFAULT_N_DOCS,
        help=f"Number of documents to generate (default: {DEFAULT_N_DOCS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = generate_docs(n_docs=args.n_docs, seed=args.seed)
    df = pd.DataFrame([asdict(d) for d in docs])
    df.to_parquet(output_path, index=False)

    print(f"Generated {len(df)} patient education docs.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()