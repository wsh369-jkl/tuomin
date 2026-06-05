"""Compatibility aliases for the legacy review namespace."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api import assistant
from app.schemas.desensitize import (
    AssistantResult,
    AssistantTaskStatus,
    ReviewGenerateRequest,
)

router = APIRouter(prefix="/review", tags=["review"])

# Legacy callers may still import this name in tests or helper code.
review_tasks = assistant.assistant_tasks


@router.post("/upload", response_model=AssistantTaskStatus)
async def upload_for_review(
    file: UploadFile = File(...),
    use_llm: bool = True,
    use_custom: bool = True,
    llm_model: Optional[str] = None,
    anonymization_strategy: Optional[str] = None,
    async_mode: bool = True,
):
    del use_llm, use_custom, anonymization_strategy, async_mode
    return await assistant.upload_for_assistant(file=file, llm_model=llm_model)


@router.post("/generate", response_model=AssistantTaskStatus)
async def generate_review(request: ReviewGenerateRequest):
    existing_task = assistant.assistant_tasks.get(request.task_id)
    if existing_task is not None:
        return assistant._assistant_status_response(request.task_id, existing_task)

    raise HTTPException(
        status_code=410,
        detail=(
            "旧版两段式 review/generate 工作流已下线。"
            "请改用 /api/v1/assistant/upload 或 /api/v1/review/upload 直接启动独立律师辅助分析。"
        ),
    )


@router.get("/status/{review_id}", response_model=AssistantTaskStatus)
async def get_review_status(review_id: str):
    return await assistant.get_assistant_status(review_id)


@router.get("/result/{review_id}", response_model=AssistantResult)
async def get_review_result(review_id: str):
    return await assistant.get_assistant_result(review_id)
