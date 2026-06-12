from __future__ import annotations

import os
import random
import time
from typing import Any

import gradio as gr

from src.model_service import get_form_options, get_model, predict_addicted_score


REQUIRED_FIELDS = {
    "age": "Idade",
    "gender": "Gênero",
    "academic_level": "Nível acadêmico",
    "country": "País",
    "avg_daily_usage_hours": "Horas diárias em redes sociais",
    "most_used_platform": "Plataforma mais usada",
    "affects_academic_performance": "Afeta o desempenho acadêmico",
    "sleep_hours_per_night": "Horas de sono por noite",
    "mental_health_score": "Pontuação de saúde mental",
    "relationship_status": "Status de relacionamento",
    "conflicts_over_social_media": "Conflitos por redes sociais",
}


OPTION_LABELS = {
    "Gender": {
        "Female": "Feminino",
        "Male": "Masculino",
    },
    "Academic_Level": {
        "Graduate": "Pós-graduação",
        "High School": "Ensino médio",
        "Undergraduate": "Graduação",
    },
    "Affects_Academic_Performance": {
        "No": "Não",
        "Yes": "Sim",
    },
    "Relationship_Status": {
        "Complicated": "Complicado",
        "In Relationship": "Em relacionamento",
        "Single": "Solteiro",
    },
}


APP_CSS = """
:root {
    --color-bg: #111111;
    --color-panel: #111111;
    --color-text: #ffffff;
    --color-muted: #d8d8d8;
    --color-border: #ffffff;
    --color-error: #b42318;
}

.gradio-container {
    background: var(--color-bg) !important;
    color: var(--color-text) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#app-shell {
    max-width: 1120px;
    margin: 0 auto;
    padding: 28px 16px 36px;
}

#app-title h1 {
    color: var(--color-text);
    font-size: 34px;
    line-height: 1.1;
    letter-spacing: 0;
    margin-bottom: 6px;
    text-align: center;
}

#app-title p {
    color: var(--color-muted);
    font-size: 15px;
    margin-top: 0;
    text-align: center;
}

.form-panel {
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 18px;
}

#result-screen {
    min-height: 62vh;
    align-items: center;
    justify-content: center;
    display: flex;
}

.result-content {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.primary-button {
    background: #111111 !important;
    border: 1px solid #111111 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

.secondary-button {
    background: #ffffff !important;
    border: 1px solid var(--color-border) !important;
    color: #111111 !important;
    border-radius: 6px !important;
}

.result-card {
    border-radius: 8px;
    padding: 34px;
    width: min(780px, 100%);
    min-height: 260px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    border: 1px solid var(--color-border);
    background: #111111;
}

.result-card strong {
    color: var(--color-text);
    font-size: 28px;
    line-height: 1.18;
    max-width: 100%;
    text-align: center;
    white-space: nowrap;
}

.score-value {
    color: var(--color-text) !important;
    display: block;
    font-size: 42px !important;
    font-weight: 800;
    line-height: 1;
}

.score-scale {
    color: var(--color-muted) !important;
    display: block;
    font-size: 13px !important;
}

.result-card span,
.result-card li {
    color: var(--color-muted);
    font-size: 14px;
}

.result-card.error {
    border-color: var(--color-error);
}

.result-card.error strong,
.result-card.error li {
    color: var(--color-error);
}

.recommendation-card {
    margin-top: 18px;
    width: min(780px, 100%);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 18px;
    background: #111111;
}

.recommendation-card strong {
    color: var(--color-text);
    display: block;
    font-size: 18px;
    margin-bottom: 8px;
}

.recommendation-card p,
.recommendation-card li {
    color: var(--color-muted);
    font-size: 14px;
    line-height: 1.45;
}

.recommendation-card ul {
    margin: 8px 0 0;
    padding-left: 18px;
}

@media (max-width: 760px) {
    .result-card strong {
        white-space: normal;
    }
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: .08em;
    font-size: 12px !important;
    color: var(--color-muted);
}

footer {
    display: none !important;
}
"""


def translated_choices(field: str, options: dict[str, list[Any]]) -> list[tuple[str, Any]]:
    labels = OPTION_LABELS.get(field, {})
    return [(labels.get(option, str(option)), option) for option in options[field]]


def describe_addiction_level(predicted_class: int) -> str:
    if predicted_class == 3:
        return "Uso com baixo risco de dependência em redes sociais"
    if predicted_class in (4, 5):
        return "Sinais leves de dependência em redes sociais"
    if predicted_class == 6:
        return "Risco moderado de dependência em redes sociais"
    if predicted_class in (7, 8):
        return "Risco alto de dependência em redes sociais"
    if predicted_class == 9:
        return "Risco muito alto de dependência em redes sociais"

    return f"Nível de dependência {predicted_class}"


def build_recommendation_text(predicted_class: int, values: dict[str, Any]) -> str:
    if predicted_class < 7:
        return (
            "<div class='recommendation-card'>"
            "<strong>Como manter esse resultado sob controle</strong>"
            "<p>Seu resultado não ficou na faixa de maior atenção. Mesmo assim, vale manter uma rotina "
            "equilibrada: usar redes sociais com horários definidos, preservar o sono e fazer pausas longe da tela.</p>"
            "</div>"
        )

    factors = [
        (
            "conflitos por redes sociais",
            float(values["conflicts_over_social_media"]) / 5,
            "evite responder no impulso, silencie notificações em momentos de tensão e defina limites para conversas que geram desgaste.",
        ),
        (
            "tempo diário em redes sociais",
            min(float(values["avg_daily_usage_hours"]) / 8, 1.5),
            "crie um limite diário de uso, desative notificações não essenciais e separe períodos do dia sem celular.",
        ),
        (
            "poucas horas de sono",
            max((7 - float(values["sleep_hours_per_night"])) / 7, 0),
            "evite redes sociais antes de dormir e tente manter um horário mais regular para descansar.",
        ),
        (
            "pontuação de saúde mental baixa",
            max((6 - float(values["mental_health_score"])) / 6, 0),
            "inclua pausas reais, atividades fora da tela e converse com alguém de confiança se isso estiver afetando sua rotina.",
        ),
    ]

    if values["affects_academic_performance"] == "Yes":
        factors.append(
            (
                "impacto no desempenho acadêmico",
                0.85,
                "deixe as redes sociais bloqueadas durante o estudo e trabalhe com metas curtas de foco e intervalos planejados.",
            )
        )

    main_factor, _, main_action = max(factors, key=lambda item: item[1])
    secondary_factors = sorted(factors, key=lambda item: item[1], reverse=True)[1:3]
    secondary_items = "".join(
        f"<li>{factor}: {action}</li>" for factor, _, action in secondary_factors
    )

    return (
        "<div class='recommendation-card'>"
        "<strong>Principal ponto de atenção</strong>"
        f"<p>O fator que mais chama atenção neste resultado é <b>{main_factor}</b>. "
        f"Para reduzir o risco, uma boa primeira ação é: {main_action}</p>"
        "<ul>"
        f"{secondary_items}"
        "</ul>"
        "</div>"
    )


def validate_inputs(values: dict[str, Any]) -> list[str]:
    missing_fields = [
        label
        for field, label in REQUIRED_FIELDS.items()
        if values.get(field) is None or values.get(field) == ""
    ]

    if missing_fields:
        return [f"Preencha: {', '.join(missing_fields)}."]

    validation_errors = []
    if values["avg_daily_usage_hours"] < 0:
        validation_errors.append("As horas diárias não podem ser negativas.")
    if values["sleep_hours_per_night"] < 0:
        validation_errors.append("As horas de sono não podem ser negativas.")
    if not 0 <= values["mental_health_score"] <= 10:
        validation_errors.append("A saúde mental deve estar entre 0 e 10.")
    if not 0 <= values["conflicts_over_social_media"] <= 5:
        validation_errors.append("Os conflitos devem estar entre 0 e 5.")

    return validation_errors


def generate_random_form_values() -> tuple[Any, ...]:
    form_options = get_form_options()
    return (
        random.randint(12, 60),
        random.choice(form_options["Gender"]),
        random.choice(form_options["Academic_Level"]),
        random.choice(form_options["Country"]),
        round(random.uniform(0, 16), 1),
        random.choice(form_options["Most_Used_Platform"]),
        random.choice(form_options["Affects_Academic_Performance"]),
        round(random.uniform(0, 12), 1),
        random.randint(0, 10),
        random.choice(form_options["Relationship_Status"]),
        random.randint(0, 5),
    )


def classify_student(
    age: float | None,
    gender: str | None,
    academic_level: str | None,
    country: str | None,
    avg_daily_usage_hours: float | None,
    most_used_platform: str | None,
    affects_academic_performance: str | None,
    sleep_hours_per_night: float | None,
    mental_health_score: float | None,
    relationship_status: str | None,
    conflicts_over_social_media: float | None,
) -> tuple[str, Any, Any]:
    form_values = {
        "age": age,
        "gender": gender,
        "academic_level": academic_level,
        "country": country,
        "avg_daily_usage_hours": avg_daily_usage_hours,
        "most_used_platform": most_used_platform,
        "affects_academic_performance": affects_academic_performance,
        "sleep_hours_per_night": sleep_hours_per_night,
        "mental_health_score": mental_health_score,
        "relationship_status": relationship_status,
        "conflicts_over_social_media": conflicts_over_social_media,
    }

    validation_errors = validate_inputs(form_values)
    if validation_errors:
        error_list = "".join(f"<li>{error}</li>" for error in validation_errors)
        return (
            f"<div class='result-card error'><strong>Informações pendentes</strong>"
            f"<ul>{error_list}</ul></div>",
            gr.update(visible=True),
            gr.update(visible=True),
        )

    input_data = {
        "Age": int(age),
        "Gender": gender,
        "Academic_Level": academic_level,
        "Country": country,
        "Avg_Daily_Usage_Hours": float(avg_daily_usage_hours),
        "Most_Used_Platform": most_used_platform,
        "Sleep_Hours_Per_Night": float(sleep_hours_per_night),
        "Mental_Health_Score": int(mental_health_score),
        "Relationship_Status": relationship_status,
        "Conflicts_Over_Social_Media": int(conflicts_over_social_media),
    }

    result = predict_addicted_score(input_data)
    result_description = describe_addiction_level(result.predicted_class)
    recommendation_text = build_recommendation_text(result.predicted_class, form_values)
    confidence_text = ""
    if result.confidence is not None:
        confidence_text = f"<span>Confiança estimada: {result.confidence:.1%}</span>"

    return (
        "<div class='result-content'>"
        "<div class='result-card success'>"
        "<span class='eyebrow'>Nível de vício</span>"
        f"<span class='score-value'>{result.predicted_class}</span>"
        "<span class='score-scale'>Escala possível: mínimo 3 e máximo 9</span>"
        f"<strong>{result_description}</strong>"
        f"{confidence_text}"
        "</div>"
        f"{recommendation_text}"
        "</div>",
        gr.update(visible=False),
        gr.update(visible=True),
    )


def build_interface() -> gr.Blocks:
    form_options = get_form_options()
    get_model()

    with gr.Blocks(title="Classificador de Vício em Redes Sociais") as demo:
        with gr.Column(elem_id="app-shell"):
            gr.Markdown(
                """
                # Classificador de Vício em Redes Sociais
                Verifique seu nível de vício em redes sociais
                """,
                elem_id="app-title",
            )

            with gr.Column(elem_classes=["form-panel"]) as form_section:
                age = gr.Number(label="Idade", precision=0)
                gender = gr.Dropdown(
                    label="Gênero",
                    choices=translated_choices("Gender", form_options),
                    value=None,
                )
                academic_level = gr.Dropdown(
                    label="Nível acadêmico",
                    choices=translated_choices("Academic_Level", form_options),
                    value=None,
                )
                country = gr.Dropdown(
                    label="País",
                    choices=form_options["Country"],
                    value=None,
                    filterable=True,
                )
                avg_daily_usage_hours = gr.Number(
                    label="Horas diárias em redes sociais",
                )
                most_used_platform = gr.Dropdown(
                    label="Plataforma mais usada",
                    choices=form_options["Most_Used_Platform"],
                    value=None,
                )
                affects_academic_performance = gr.Dropdown(
                    label="Afeta o desempenho acadêmico",
                    choices=translated_choices("Affects_Academic_Performance", form_options),
                    value=None,
                )
                sleep_hours_per_night = gr.Number(
                    label="Horas de sono por noite",
                )
                mental_health_score = gr.Number(
                    label="Pontuação de saúde mental",
                    precision=0,
                    minimum=0,
                    maximum=10,
                )
                relationship_status = gr.Dropdown(
                    label="Status de relacionamento",
                    choices=translated_choices("Relationship_Status", form_options),
                    value=None,
                )
                conflicts_over_social_media = gr.Number(
                    label="Conflitos por redes sociais",
                    precision=0,
                    minimum=0,
                    maximum=5,
                )

                with gr.Row():
                    random_button = gr.Button(
                        "Preencher aleatório",
                        elem_classes=["secondary-button"],
                    )
                    submit_button = gr.Button(
                        "Classificar",
                        elem_classes=["primary-button"],
                    )
                    gr.ClearButton(
                        value="Limpar",
                        components=[
                            age,
                            gender,
                            academic_level,
                            country,
                            avg_daily_usage_hours,
                            most_used_platform,
                            affects_academic_performance,
                            sleep_hours_per_night,
                            mental_health_score,
                            relationship_status,
                            conflicts_over_social_media,
                        ],
                        elem_classes=["secondary-button"],
                    )

            with gr.Column(visible=False, elem_id="result-screen") as result_section:
                result_output = gr.HTML(
                    "<div class='result-card'><span class='eyebrow'>Resultado</span>"
                    "<span>O nível de vício previsto aparecerá aqui após o envio.</span></div>"
                )

            inputs = [
                age,
                gender,
                academic_level,
                country,
                avg_daily_usage_hours,
                most_used_platform,
                affects_academic_performance,
                sleep_hours_per_night,
                mental_health_score,
                relationship_status,
                conflicts_over_social_media,
            ]

            random_button.click(
                fn=generate_random_form_values,
                inputs=None,
                outputs=inputs,
            )

            submit_button.click(
                fn=classify_student,
                inputs=inputs,
                outputs=[result_output, form_section, result_section],
            )

    return demo


demo = build_interface()


def resolve_server_name() -> str:
    if os.getenv("HOST"):
        return os.getenv("HOST", "127.0.0.1")

    deploy_environment_markers = [
        "SPACE_ID",
        "WEBSITE_SITE_NAME",
        "WEBSITES_PORT",
        "AZURE_HTTP_USER_AGENT",
    ]
    if any(os.getenv(marker) for marker in deploy_environment_markers):
        return "0.0.0.0"

    return "127.0.0.1"


def should_use_fixed_port() -> bool:
    return bool(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or os.getenv("HOST"))


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    server_name = resolve_server_name()
    server_port = port if should_use_fixed_port() else None
    port_label = str(server_port) if server_port is not None else "auto"
    print(f"Iniciando Gradio em {server_name}:{port_label}", flush=True)
    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True,
        show_error=True,
        footer_links=[],
        ssr_mode=False,
        css=APP_CSS,
    )
    print("Servidor Gradio iniciado.", flush=True)
    while True:
        time.sleep(60)
