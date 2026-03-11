"""Template-related APIs kept under the historical route for compatibility."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.history import ConfigTemplate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])
TEMPLATE_READ_FAILURE_DETAIL = "模板读取失败，请稍后重试。"
TEMPLATE_CREATE_FAILURE_DETAIL = "模板保存失败，请稍后重试。"
TEMPLATE_DELETE_FAILURE_DETAIL = "模板删除失败，请稍后重试。"


class TemplateCreateRequest(BaseModel):
    """Payload for saving a reusable configuration template."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    config_data: Dict[str, Any]
    is_default: bool = False


@router.get("/templates")
async def get_templates(db: Session = Depends(get_db)):
    """Return all saved configuration templates."""
    try:
        templates = db.query(ConfigTemplate).order_by(ConfigTemplate.created_at.desc()).all()
        return [template.to_dict() for template in templates]
    except Exception as exc:
        logger.error("Failed to load templates: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=TEMPLATE_READ_FAILURE_DETAIL)


@router.post("/templates")
async def create_template(
    request: TemplateCreateRequest,
    db: Session = Depends(get_db),
):
    """Create a reusable configuration template."""
    try:
        existing = db.query(ConfigTemplate).filter(ConfigTemplate.name == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Template name already exists.")

        if request.is_default:
            db.query(ConfigTemplate).update({"is_default": 0})

        template = ConfigTemplate(
            name=request.name,
            description=request.description,
            config_data=request.config_data,
            is_default=1 if request.is_default else 0,
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        return template.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create template: %s", exc, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=TEMPLATE_CREATE_FAILURE_DETAIL)


@router.delete("/templates/{template_id}")
async def delete_template(template_id: int, db: Session = Depends(get_db)):
    """Delete a reusable configuration template."""
    try:
        template = db.query(ConfigTemplate).filter(ConfigTemplate.id == template_id).first()
        if template is None:
            raise HTTPException(status_code=404, detail="Template was not found.")

        db.delete(template)
        db.commit()
        return {"status": "success", "message": "Template deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete template: %s", exc, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=TEMPLATE_DELETE_FAILURE_DETAIL)
