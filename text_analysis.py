#!/usr/bin/env python3
"""
Simple Gender Stereotype Analysis
Outputs a single comprehensive paragraph for each movie text.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import spacy
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
import torch

# ---------------------------
# Configuration
# ---------------------------
CANDIDATE_LABELS = [
    "profession_present",
    "aspiration_present",
    "dependent_relation",
    "objectification",
    "power_attribute",
    "emotional_description",
    "physical_appearance"
]

HYPOTHESIS_TEMPLATE = "This sentence is about {}."

DEPENDENT_PATTERNS = [
    r"\b(daughter of|son of|wife of|husband of|child of|born to|married to|widow of|widower of)\b",
    r"\b(belongs to|owned by|property of)\b"
]

OBJECTIFICATION_KEYWORDS = {"seductive", "sexy", "beautiful", "gorgeous", "stunning"}
EMOTIONAL_KEYWORDS = {"emotional", "crying", "fragile", "vulnerable"}
POWER_KEYWORDS = {"strong", "leader", "independent", "confident"}
APPEARANCE_WORDS = {"beautiful", "pretty", "looks", "appearance"}

MALE_PRONOUNS = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}

# ---------------------------
# Helpers
# ---------------------------
def find_text_files(repo_dir: Path):
    repo_dir = Path(repo_dir)
    return [Path(root) / f for root, _, files in os.walk(repo_dir) for f in files if f.endswith(".txt")]

def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except:
        return path.read_text(encoding="latin-1")

def prepare_spacy():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
    return nlp

def init_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

def detect_gender(sentence: str):
    words = sentence.lower().split()
    if sum(w in FEMALE_PRONOUNS for w in words) > sum(w in MALE_PRONOUNS for w in words):
        return "female"
    elif sum(w in MALE_PRONOUNS for w in words):
        return "male"
    return "neutral"

def heuristic_flags(sentence: str):
    s = sentence.lower()
    flags = {}
    if any(re.search(pat, s) for pat in DEPENDENT_PATTERNS):
        flags["dependent_relation"] = True
    if any(w in s for w in OBJECTIFICATION_KEYWORDS):
        flags["objectification"] = True
    if any(w in s for w in EMOTIONAL_KEYWORDS):
        flags["emotional_description"] = True
    if any(w in s for w in POWER_KEYWORDS):
        flags["power_attribute"] = True
    if any(w in s for w in APPEARANCE_WORDS):
        flags["physical_appearance"] = True
    return flags

# ---------------------------
# Analysis
# ---------------------------
def generate_paragraph(movie_id, counts, total_sentences, severity, score):
    paragraph = [f"**Gender Stereotype Analysis for '{movie_id}'** – Severity Level: {severity} ({score:.1f}/100). "]
    flagged = sum(counts.values())
    paragraph.append(f"Out of {total_sentences} sentences analyzed, {flagged} contained stereotype indicators. ")

    issues = []
    if counts.get("dependent_relation", 0):
        issues.append("Characters are defined mainly by relationships (e.g., 'wife of', 'daughter of').")
    if counts.get("objectification", 0):
        issues.append("Women are described in sexualized or appearance-focused terms.")
    if counts.get("emotional_description", 0):
        issues.append("Characters are portrayed as overly emotional or fragile.")
    if counts.get("physical_appearance", 0):
        issues.append("Excessive focus on physical looks over personality or skills.")
    if counts.get("profession_present", 0) < total_sentences * 0.1:
        issues.append("Very limited mention of careers or professions.")
    if counts.get("aspiration_present", 0) < total_sentences * 0.05:
        issues.append("Lack of personal goals or ambitions.")

    if issues:
        paragraph.append("**Stereotypes Identified:** " + "; ".join(issues) + " ")
    else:
        paragraph.append("Balanced representation with minimal gender stereotypes. ")

    # Suggestions
    suggestions = [
        "Give characters independent storylines and goals",
        "Reduce appearance-based descriptions, focus on skills and personality",
        "Show diverse professional roles and ambitions for all characters"
    ]
    paragraph.append("**Suggestions:** " + "; ".join(suggestions) + ".")
    return "".join(paragraph)

def analyze_text(movie_id, text, nlp, zs):
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    total = len(sentences)
    counts = {label: 0 for label in CANDIDATE_LABELS}

    for sent in tqdm(sentences, desc=f"Analyzing {movie_id}", leave=False):
        heur = heuristic_flags(sent)
        for h in heur:
            counts[h] += 1
        try:
            res = zs(sent, CANDIDATE_LABELS, hypothesis_template=HYPOTHESIS_TEMPLATE, multi_label=True)
            for lbl, score in zip(res["labels"], res["scores"]):
                if score >= 0.6:
                    counts[lbl] += 1
        except:
            pass

    severity, score = calculate_severity(counts, total)
    return generate_paragraph(movie_id, counts, total, severity, score)

def calculate_severity(counts, total):
    weights = {
        "dependent_relation": 3.0,
        "objectification": 3.0,
        "emotional_description": 2.0,
        "physical_appearance": 2.0,
        "profession_present": -1.0,
        "aspiration_present": -1.0,
        "power_attribute": -0.5
    }
    severity_score = 0.0
    for label, count in counts.items():
        weight = weights.get(label, 1.0)
        severity_score += (count / max(1, total)) * weight
    severity_score = max(0, min(100, (severity_score + 2) * 25))
    if severity_score >= 75:
        return "HIGH", severity_score
    elif severity_score >= 50:
        return "MODERATE", severity_score
    elif severity_score >= 25:
        return "LOW", severity_score
    return "MINIMAL", severity_score

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default="./movies")
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = find_text_files(Path(args.repo_dir))
    nlp = prepare_spacy()
    zs = init_pipeline()

    for f in files:
        movie_id = f.stem
        text = load_text(f)
        if not text.strip():
            continue
        paragraph = analyze_text(movie_id, text, nlp, zs)
        out_path = Path(args.out_dir) / f"{movie_id}_paragraph.txt"
        out_path.write_text(paragraph, encoding="utf-8")
        print(f"✓ {movie_id} -> {out_path}")

if __name__ == "__main__":
    main()
