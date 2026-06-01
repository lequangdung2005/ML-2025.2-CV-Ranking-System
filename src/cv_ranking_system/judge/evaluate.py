from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from cv_ranking_system import config
from cv_ranking_system.judge.schemas import Judgment
from cv_ranking_system.llms.openai import OpenAIClient, RetryConfig
from cv_ranking_system.utils.utils import provider_api_key, provider_base_url, text_model

logger = logging.getLogger(__name__)


def _judge_prompt(*, jd_text: str, cv_json: str) -> str:
    return (
        "You are an objective resume evaluator. Given a job description and a candidate CV JSON, "
        "produce a strict JSON judgment with this schema:\n"
        "{score_0_100:int, strengths:[{statement,evidence{references[],quotes[]}}], "
        "weaknesses:[...], must_haves_met:[...], must_haves_missing:[...]}\n"
        "Rules: Return JSON only. Score must be an integer 0-100. "
        "Evidence.references should point to CV JSON fields like 'skills[2]' "
        "or 'experience[0].highlights[1]'. "
        "Evidence.quotes must be verbatim snippets from the CV JSON content. "
        "Do not invent facts.\n\n"
        f"JOB_DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE_CV_JSON:\n{cv_json}"
    )


def evaluate_candidate(*, jd_text: str, cv: dict, trace_id: str) -> Judgment:
    model = text_model()
    if not model:
        raise ValueError("Missing TEXT model: set config.TEXT_MODEL")

    oai = OpenAIClient(
        base_url=provider_base_url(),
        api_key=provider_api_key(),
        timeout_s=config.CVRS_HTTP_TIMEOUT_S,
        retry=RetryConfig(
            max_retries=config.CVRS_HTTP_MAX_RETRIES,
            initial_backoff_s=config.CVRS_HTTP_INITIAL_BACKOFF_S,
            max_backoff_s=config.CVRS_HTTP_MAX_BACKOFF_S,
        ),
    )

    cv_json = json.dumps(cv, ensure_ascii=True, indent=2)
    raw = oai.chat_text(
        prompt=_judge_prompt(jd_text=jd_text, cv_json=cv_json), model=model, trace_id=trace_id
    )
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge did not return valid JSON: {e}") from e
    try:
        j = Judgment.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"Judge output failed schema validation: {e}") from e
    # Clamp score safety.
    if j.score_0_100 < 0 or j.score_0_100 > 100:
        raise ValueError("Judge score_0_100 out of range")
    return j
