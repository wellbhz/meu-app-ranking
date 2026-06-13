from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "addicted_score_svm.joblib"

MODEL_FEATURES = [
    "Age",
    "Gender",
    "Academic_Level",
    "Country",
    "Avg_Daily_Usage_Hours",
    "Most_Used_Platform",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Relationship_Status",
    "Conflicts_Over_Social_Media",
]

FORM_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Academic_Level": ["Graduate", "High School", "Undergraduate"],
    "Country": [
        "Afghanistan",
        "Albania",
        "Andorra",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Bhutan",
        "Bolivia",
        "Bosnia",
        "Brazil",
        "Bulgaria",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Ecuador",
        "Egypt",
        "Estonia",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Greece",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kuwait",
        "Latvia",
        "Lebanon",
        "Lithuania",
        "Luxembourg",
        "Malaysia",
        "Maldives",
        "Mexico",
        "Moldova",
        "Monaco",
        "Mongolia",
        "Morocco",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "Norway",
        "Oman",
        "Pakistan",
        "Paraguay",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Russia",
        "Saudi Arabia",
        "Serbia",
        "Singapore",
        "Slovakia",
        "Slovenia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sri Lanka",
        "Sweden",
        "Switzerland",
        "Syria",
        "Taiwan",
        "Thailand",
        "Turkey",
        "Ukraine",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
        "Uruguay",
        "Uzbekistan",
        "Venezuela",
        "Vietnam",
        "Yemen",
    ],
    "Most_Used_Platform": [
        "Facebook",
        "Instagram",
        "KakaoTalk",
        "LINE",
        "LinkedIn",
        "Snapchat",
        "TikTok",
        "Twitter",
        "VKontakte",
        "WeChat",
        "WhatsApp",
        "YouTube",
    ],
    "Affects_Academic_Performance": ["No", "Yes"],
    "Relationship_Status": ["Complicated", "In Relationship", "Single"],
}


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    confidence: float | None
    probabilities: dict[int, float]


def get_form_options() -> dict[str, list[Any]]:
    return FORM_OPTIONS


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Gere ou adicione o arquivo "
            f"{MODEL_PATH} antes de iniciar a aplicacao."
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    return load_model()


def normalize_model_input(input_data: dict[str, Any]) -> pd.DataFrame:
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if input_data.get(feature) is None or input_data.get(feature) == ""
    ]
    if missing_features:
        raise ValueError(f"Campos ausentes para o modelo: {', '.join(missing_features)}")

    return pd.DataFrame([{feature: input_data[feature] for feature in MODEL_FEATURES}])


def predict_addicted_score(input_data: dict[str, Any]) -> PredictionResult:
    model = get_model()
    sample = normalize_model_input(input_data)

    predicted_class = int(model.predict(sample)[0])
    probabilities: dict[int, float] = {}
    confidence = None

    classifier = model.named_steps.get("model") if hasattr(model, "named_steps") else None

    if classifier is not None and hasattr(classifier, "classes_"):
        try:
            probability_values = model.predict_proba(sample)[0]
            probabilities = {
                int(class_label): round(float(probability), 4)
                for class_label, probability in zip(classifier.classes_, probability_values)
            }
            confidence = probabilities.get(predicted_class)
        except Exception:
            probabilities = {}
            confidence = None

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
    )