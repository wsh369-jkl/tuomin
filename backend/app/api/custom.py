"""Custom rule management API."""
from typing import List
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.custom_recognizer_service import CustomRecognizerService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/custom", tags=["custom"])
custom_service = CustomRecognizerService()


class AddKeywordRequest(BaseModel):
    entity_type: str
    keywords: List[str]
    score: float = 0.95
    description: str = ""


class AddPatternRequest(BaseModel):
    entity_type: str
    regex: str
    context: List[str] = Field(default_factory=list)
    score: float = 0.9
    description: str = ""


@router.get("/config")
async def get_config():
    """Return the current custom keyword and pattern configuration."""
    try:
        custom_service.load_config()
        return {
            "keywords": custom_service.custom_keywords,
            "patterns": custom_service.custom_patterns,
            "keywords_count": len(custom_service.custom_keywords),
            "patterns_count": len(custom_service.custom_patterns),
        }
    except Exception as exc:
        logger.error("Failed to load custom configuration: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/keywords")
async def add_keywords(request: AddKeywordRequest):
    """Create or replace a keyword-based custom rule."""
    try:
        custom_service.add_keyword_rule(
            entity_type=request.entity_type,
            keywords=request.keywords,
            score=request.score,
            description=request.description,
        )
        custom_service.save_config()
        return {
            "status": "success",
            "message": f"已保存 {request.entity_type} 关键词规则",
        }
    except Exception as exc:
        logger.error("Failed to save keyword rule: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/patterns")
async def add_pattern(request: AddPatternRequest):
    """Create or replace a regex-based custom rule."""
    try:
        custom_service.add_pattern_rule(
            entity_type=request.entity_type,
            regex=request.regex,
            context=request.context,
            score=request.score,
            description=request.description,
        )
        custom_service.save_config()
        return {
            "status": "success",
            "message": f"已保存 {request.entity_type} 正则规则",
        }
    except Exception as exc:
        logger.error("Failed to save pattern rule: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/keywords/{entity_type}")
async def delete_keywords(entity_type: str):
    """Delete all keywords for the given entity type."""
    try:
        if not custom_service.delete_keyword_rule(entity_type):
            raise HTTPException(status_code=404, detail="关键词规则不存在")

        custom_service.save_config()
        return {
            "status": "success",
            "message": f"已删除 {entity_type} 关键词规则",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete keyword rule: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/patterns/{entity_type}")
async def delete_pattern(entity_type: str):
    """Delete the regex rule for the given entity type."""
    try:
        if not custom_service.delete_pattern_rule(entity_type):
            raise HTTPException(status_code=404, detail="正则规则不存在")

        custom_service.save_config()
        return {
            "status": "success",
            "message": f"已删除 {entity_type} 正则规则",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete pattern rule: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reload")
async def reload_config():
    """Reload custom rules from disk."""
    try:
        custom_service.load_config()
        return {
            "status": "success",
            "message": "配置已重新加载",
            "keywords_count": len(custom_service.custom_keywords),
            "patterns_count": len(custom_service.custom_patterns),
        }
    except Exception as exc:
        logger.error("Failed to reload custom configuration: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/test")
async def test_custom_recognizer(text: str):
    """Run custom rule recognition against the supplied text."""
    try:
        custom_service.load_config()
        entities = custom_service.match_all(text)
        return {
            "text": text,
            "entities": entities,
            "count": len(entities),
        }
    except Exception as exc:
        logger.error("Failed to test custom recognizer: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
