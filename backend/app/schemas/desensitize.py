"""Pydantic schemas for the desensitization APIs."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    type: str = Field(..., description="Entity type")
    text: str = Field(..., description="Matched text")
    start: int = Field(..., description="Start offset")
    end: int = Field(..., description="End offset")
    score: float = Field(default=0.9, description="Confidence score")
    source: str = Field(default="unknown", description="Recognizer source")
    replacement: Optional[str] = Field(default=None, description="Prepared replacement text")
    replacement_method: Optional[str] = Field(default=None, description="Replacement generation method")
    context_label: Optional[str] = Field(default=None, description="Nearby field label")
    context_role: Optional[str] = Field(default=None, description="Inferred contract role")
    canonical_key: Optional[str] = Field(default=None, description="LLM-resolved canonical identity key")
    canonical_role: Optional[str] = Field(default=None, description="LLM-resolved canonical role")


class AnalyzeResponse(BaseModel):
    task_id: str = Field(..., description="Task ID")
    filename: str = Field(..., description="Original filename")
    text: str = Field(..., description="Extracted source text")
    entities: List[Entity] = Field(default_factory=list, description="Detected entities")
    statistics: Dict = Field(default_factory=dict, description="Entity statistics")
    metadata: Dict = Field(default_factory=dict, description="Document metadata")
    llm_model: Optional[str] = Field(default=None, description="LLM model used during analysis")
    llm_strategy: Optional[str] = Field(default=None, description="LLM strategy key used during analysis")
    llm_strategy_label: Optional[str] = Field(default=None, description="LLM strategy label used during analysis")
    anonymization_strategy: Optional[str] = Field(default=None, description="Replacement strategy key")
    anonymization_strategy_label: Optional[str] = Field(default=None, description="Replacement strategy label")


class DesensitizeRequest(BaseModel):
    task_id: str = Field(..., description="Task ID")
    entities: List[Entity] = Field(..., description="Entities confirmed by the user")
    config: Optional[Dict] = Field(default=None, description="Operator configuration")
    llm_model: Optional[str] = Field(default=None, description="Requested LLM model")
    anonymization_strategy: Optional[str] = Field(default=None, description="Requested replacement strategy")


class DesensitizeResponse(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    anonymized_text: Optional[str] = Field(default=None, description="Preview text")
    entities: List[Entity] = Field(default_factory=list, description="Entities with prepared replacements")
    metadata: Dict = Field(default_factory=dict, description="Processing metadata")
    download_url: Optional[str] = Field(default=None, description="Download URL")
    output_filename: Optional[str] = Field(default=None, description="Generated file name")
    output_file_type: Optional[str] = Field(default=None, description="Generated file type")
    preserves_format: bool = Field(default=False, description="Whether the original format was preserved")
    llm_assisted: bool = Field(default=False, description="Whether the LLM participated in processing")
    llm_model: Optional[str] = Field(default=None, description="LLM model used during processing")
    llm_strategy: Optional[str] = Field(default=None, description="LLM strategy key used during processing")
    llm_strategy_label: Optional[str] = Field(default=None, description="LLM strategy label used during processing")
    anonymization_strategy: Optional[str] = Field(default=None, description="Replacement strategy key")
    anonymization_strategy_label: Optional[str] = Field(default=None, description="Replacement strategy label")
    warning: Optional[str] = Field(default=None, description="Export warning")
    message: Optional[str] = Field(default=None, description="Status message")


class TaskStatus(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: int = Field(default=0, description="Progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")


class LLMModelOption(BaseModel):
    name: str = Field(..., description="Model name")
    installed: bool = Field(default=False, description="Whether the model is installed locally")
    is_default: bool = Field(default=False, description="Whether this model is the backend default")
    strategy_key: str = Field(..., description="Strategy key applied to this model")
    strategy_label: str = Field(..., description="Strategy label applied to this model")
    strategy_description: str = Field(..., description="Strategy description applied to this model")


class LLMModelListResponse(BaseModel):
    backend: str = Field(..., description="Configured LLM backend")
    default_model: Optional[str] = Field(default=None, description="Backend default LLM model")
    service_available: bool = Field(default=False, description="Whether the backend service is reachable")
    models: List[LLMModelOption] = Field(default_factory=list, description="Selectable models")


class RuntimeStatusResponse(BaseModel):
    backend: str = Field(..., description="Configured LLM backend")
    platform: str = Field(..., description="Runtime platform label")
    ready: bool = Field(default=False, description="Whether the runtime is ready for full-quality processing")
    ollama_install_detected: bool = Field(default=False, description="Whether an Ollama installation was detected")
    ollama_path: Optional[str] = Field(default=None, description="Detected Ollama installation path")
    service_available: bool = Field(default=False, description="Whether the Ollama service is reachable")
    required_model: Optional[str] = Field(default=None, description="Required local model name")
    required_model_installed: bool = Field(default=False, description="Whether the required model is installed")
    default_model: Optional[str] = Field(default=None, description="Default runtime model")
    installer_hint: str = Field(default="", description="Suggested startup hint for the current platform")
    download_hint: str = Field(default="", description="Suggested model download command or script")
    recommended_action: str = Field(default="", description="Single-sentence recommended next step")
