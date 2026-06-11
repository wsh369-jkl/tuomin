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
    group_id: Optional[str] = Field(default=None, description="Derived canonical group ID")
    group_label: Optional[str] = Field(default=None, description="Human-readable group label")
    needs_review: bool = Field(default=False, description="Whether this entity should be reviewed manually")
    review_reason: Optional[str] = Field(default=None, description="Reason the entity should be reviewed")
    metadata: Dict = Field(default_factory=dict, description="Recognizer metadata and identity hints")


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
    desensitize_mode: Optional[str] = Field(default=None, description="Requested processing workflow")
    page_session_id: Optional[str] = Field(default=None, description="Frontend page session ID")
    async_mode: bool = Field(default=False, description="Whether to run anonymization/export in the background")


class DesensitizeResponse(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    anonymized_text: Optional[str] = Field(default=None, description="Preview text")
    entities: List[Entity] = Field(default_factory=list, description="Entities with prepared replacements")
    metadata: Dict = Field(default_factory=dict, description="Processing metadata")
    download_url: Optional[str] = Field(default=None, description="Download URL")
    mapping_download_url: Optional[str] = Field(default=None, description="Mapping document download URL")
    output_filename: Optional[str] = Field(default=None, description="Generated file name")
    mapping_output_filename: Optional[str] = Field(default=None, description="Generated mapping file name")
    output_file_type: Optional[str] = Field(default=None, description="Generated file type")
    mapping_output_file_type: Optional[str] = Field(default=None, description="Generated mapping file type")
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
    filename: Optional[str] = Field(default=None, description="Original filename")
    status: str = Field(..., description="Task status")
    progress: int = Field(default=0, description="Progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    error_message: Optional[str] = Field(default=None, description="Terminal error message")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")


class PdfWordAuditStatus(BaseModel):
    audit_id: str = Field(..., description="Audit task ID")
    filename: Optional[str] = Field(default=None, description="Original PDF filename")
    template_filename: Optional[str] = Field(default=None, description="WPS DOCX template filename")
    status: str = Field(..., description="Task status")
    progress: int = Field(default=0, description="Progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    error_message: Optional[str] = Field(default=None, description="Terminal error message")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")


class PdfWordAuditCorrection(BaseModel):
    id: str = Field(..., description="Correction candidate ID")
    wps_unit_id: str = Field(..., description="Matched WPS text unit ID")
    page_no: Optional[int] = Field(default=None, description="Estimated page number")
    old_text: str = Field(default="", description="WPS text")
    new_text: str = Field(default="", description="OCR suggested text")
    action: str = Field(default="review", description="Correction action")
    confidence: float = Field(default=0.0, description="OCR confidence")
    alignment_score: float = Field(default=0.0, description="Alignment score")
    reason: str = Field(default="", description="Reason for action")
    comment_text: str = Field(default="", description="DOCX comment body")
    sensitive_low_priority: bool = Field(default=False, description="Whether this likely belongs to a desensitization field")


class PdfWordAuditFinding(BaseModel):
    id: str = Field(..., description="Finding ID")
    severity: str = Field(default="medium", description="Finding severity")
    category: str = Field(default="substitution", description="Difference category")
    page_no: Optional[int] = Field(default=None, description="Page number")
    wps_text: str = Field(default="", description="WPS converted text")
    suggested_text: str = Field(default="", description="Preferred model evidence text")
    diff_ops: List[Dict] = Field(default_factory=list, description="Character/token diff operations")
    confidence: float = Field(default=0.0, description="Finding confidence")
    status: str = Field(default="suspected_error", description="Finding status")
    evidence_sources: List[str] = Field(default_factory=list, description="Evidence sources")
    bbox_refs: List[Dict] = Field(default_factory=list, description="Bounding-box references")
    crop_refs: List[str] = Field(default_factory=list, description="Crop/image references")
    wps_anchor: Dict = Field(default_factory=dict, description="WPS DOCX anchor metadata")
    reason: str = Field(default="", description="Decision reason")
    requires_human_review: bool = Field(default=True, description="Whether a human should review")


class PdfWordAuditResult(BaseModel):
    audit_id: str = Field(..., description="Audit task ID")
    filename: str = Field(..., description="Original PDF filename")
    template_filename: str = Field(..., description="WPS DOCX template filename")
    status: str = Field(default="completed", description="Task status")
    metadata: Dict = Field(default_factory=dict, description="Audit metadata")
    product_report: Dict = Field(default_factory=dict, description="Stable product-facing report summary")
    page_risk_summary: List[Dict] = Field(default_factory=list, description="Per-page risk summary")
    table_summary: Dict = Field(default_factory=dict, description="Table review summary")
    coverage_summary: Dict = Field(default_factory=dict, description="Coverage review summary")
    artifact_manifest: Dict = Field(default_factory=dict, description="Artifact manifest for reviewed DOCX/report/evidence")
    review_task_summary: Dict = Field(default_factory=dict, description="Summary of remaining review tasks")
    human_review_queue: List[Dict] = Field(default_factory=list, description="Human review queue items")
    findings: List[PdfWordAuditFinding] = Field(default_factory=list, description="Full OCR audit findings")
    corrections: List[PdfWordAuditCorrection] = Field(default_factory=list, description="Correction and review candidates")
    download_url: Optional[str] = Field(default=None, description="Audited DOCX download URL")
    report_url: Optional[str] = Field(default=None, description="Full JSON report URL")
    evidence_url: Optional[str] = Field(default=None, description="Evidence ZIP URL")
    output_filename: Optional[str] = Field(default=None, description="Audited DOCX filename")


class BatchFileItem(BaseModel):
    item_id: str = Field(..., description="Batch item ID")
    filename: str = Field(..., description="Original filename")
    relative_path: str = Field(..., description="Relative path within the selected folder")
    status: str = Field(..., description="Processing status")
    progress: int = Field(default=0, description="Item progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    error_message: Optional[str] = Field(default=None, description="Terminal error message")
    entities_count: int = Field(default=0, description="Detected entity count")
    output_filename: Optional[str] = Field(default=None, description="Generated output filename")
    mapping_output_filename: Optional[str] = Field(default=None, description="Generated mapping filename")
    output_file_type: Optional[str] = Field(default=None, description="Generated output file type")
    mapping_output_file_type: Optional[str] = Field(default=None, description="Generated mapping file type")
    preserves_format: bool = Field(default=False, description="Whether the original format was preserved")
    warning: Optional[str] = Field(default=None, description="Export warning")
    download_url: Optional[str] = Field(default=None, description="Single-file download URL")
    mapping_download_url: Optional[str] = Field(default=None, description="Mapping document download URL")
    metadata: Dict = Field(default_factory=dict, description="Per-file processing metadata")


class BatchTaskStatus(BaseModel):
    batch_id: str = Field(..., description="Batch task ID")
    folder_name: str = Field(..., description="Selected folder name")
    output_folder_name: Optional[str] = Field(default=None, description="Generated output folder name")
    status: str = Field(..., description="Batch status")
    progress: int = Field(default=0, description="Batch progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    error_message: Optional[str] = Field(default=None, description="Terminal error message")
    file_count: int = Field(default=0, description="Total file count in the batch")
    completed_count: int = Field(default=0, description="Number of finished files")
    succeeded_count: int = Field(default=0, description="Number of successful files")
    failed_count: int = Field(default=0, description="Number of failed files")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    items: List[BatchFileItem] = Field(default_factory=list, description="Per-file status entries")


class BatchResult(BaseModel):
    batch_id: str = Field(..., description="Batch task ID")
    folder_name: str = Field(..., description="Selected folder name")
    output_folder_name: Optional[str] = Field(default=None, description="Generated output folder name")
    status: str = Field(..., description="Batch status")
    progress: int = Field(default=100, description="Batch progress from 0 to 100")
    message: Optional[str] = Field(default=None, description="Status message")
    error_message: Optional[str] = Field(default=None, description="Terminal error message")
    file_count: int = Field(default=0, description="Total file count in the batch")
    completed_count: int = Field(default=0, description="Number of finished files")
    succeeded_count: int = Field(default=0, description="Number of successful files")
    failed_count: int = Field(default=0, description="Number of failed files")
    archive_download_url: Optional[str] = Field(default=None, description="ZIP archive download URL")
    archive_filename: Optional[str] = Field(default=None, description="ZIP archive filename")
    items: List[BatchFileItem] = Field(default_factory=list, description="Per-file processing results")


class LLMModelOption(BaseModel):
    name: str = Field(..., description="Model name")
    installed: bool = Field(default=False, description="Whether the model is installed locally")
    is_default: bool = Field(default=False, description="Whether this model is the backend default")
    strategy_key: str = Field(..., description="Strategy key applied to this model")
    strategy_label: str = Field(..., description="Strategy label applied to this model")
    strategy_description: str = Field(..., description="Strategy description applied to this model")
    tier: str = Field(default="standard", description="Model capability tier")
    supports_precision_review: bool = Field(default=False, description="Whether the model is suitable for review generation")
    supports_vision: bool = Field(default=False, description="Whether the model can be used for OCR/image-backed workflows")
    recommended_for: List[str] = Field(default_factory=list, description="Recommended usage scenarios")
    role: Optional[str] = Field(default=None, description="Model role in a recognition profile")
    memory_tier: Optional[str] = Field(default=None, description="Estimated memory tier")
    local_path: Optional[str] = Field(default=None, description="Resolved local model path")


class LLMModelListResponse(BaseModel):
    backend: str = Field(..., description="Configured LLM backend")
    default_model: Optional[str] = Field(default=None, description="Backend default LLM model")
    service_available: bool = Field(default=False, description="Whether the backend service is reachable")
    models: List[LLMModelOption] = Field(default_factory=list, description="Selectable models")
    profile: Optional[str] = Field(default=None, description="Recognition profile")
    primary_models: List[str] = Field(default_factory=list, description="Primary local extraction models")
    review_models: List[str] = Field(default_factory=list, description="Fragment review models")


class RuntimeStatusResponse(BaseModel):
    backend: str = Field(..., description="Configured LLM backend")
    platform: str = Field(..., description="Runtime platform label")
    ready: bool = Field(default=False, description="Whether the runtime is ready for full-quality processing")
    ollama_install_detected: bool = Field(default=False, description="Whether an Ollama installation was detected")
    ollama_path: Optional[str] = Field(default=None, description="Detected Ollama installation path")
    service_available: bool = Field(default=False, description="Whether the Ollama service is reachable")
    required_model: Optional[str] = Field(default=None, description="Required local model name")
    required_model_installed: bool = Field(default=False, description="Whether the required model is installed")
    available_processing_models: List[str] = Field(
        default_factory=list,
        description="Installed models that can be used for document processing",
    )
    preferred_processing_model: Optional[str] = Field(
        default=None,
        description="Recommended installed model for the current processing flow",
    )
    default_model: Optional[str] = Field(default=None, description="Default runtime model")
    installer_hint: str = Field(default="", description="Suggested startup hint for the current platform")
    download_hint: str = Field(default="", description="Suggested model download command or script")
    recommended_action: str = Field(default="", description="Single-sentence recommended next step")
    desensitize_mode: Optional[str] = Field(default=None, description="Configured desensitization mode")
    primary_models_ready: Optional[bool] = Field(default=None, description="Whether primary low-memory models are ready")
    review_model_installed: Optional[bool] = Field(default=None, description="Whether fragment review model is installed")
    review_model_loaded: Optional[bool] = Field(default=None, description="Whether fragment review model is currently loaded")
    estimated_memory_tier: Optional[str] = Field(default=None, description="Estimated memory tier")
    analysis_worker_process: Optional[bool] = Field(default=None, description="Whether high-memory analysis runs in one-shot workers")
    analysis_stage_isolation: Optional[bool] = Field(default=None, description="Whether high-memory stages run in sequential isolated workers")
    analysis_worker_timeout: Optional[int] = Field(default=None, description="Analysis worker timeout in seconds")


class PageSessionRequest(BaseModel):
    page_session_id: str = Field(..., description="Frontend page session ID")


class PageSessionResponse(BaseModel):
    ok: bool = Field(default=True, description="Whether the request was accepted")

