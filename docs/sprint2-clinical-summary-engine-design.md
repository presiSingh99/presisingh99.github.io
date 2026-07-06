# Sprint 2 Design: Clinical Summary Engine

## Goal

Convert a validated `PatientIntakeExtraction` object from Sprint 1 into a doctor-review draft summary.

The engine is deliberately conservative: it does not diagnose, recommend treatment, or infer facts that are not present in the validated extraction JSON. The summary is structured for review, auditability, and deterministic testing.

## Hard Constraints

- Do not diagnose.
- Do not recommend treatment.
- Do not infer facts not present in validated JSON.
- Only use extracted fields from `PatientIntakeExtraction`.
- Include missing and unresolved questions.
- Preserve source evidence references where available.
- Keep output structured and testable.
- Fail closed if required fields are missing or internally inconsistent.

## 1. Proposed Pydantic Schemas

### Core summary schemas

```python
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_path: str = Field(..., description="JSON pointer or dotted path into PatientIntakeExtraction")
    quote: str | None = Field(default=None, description="Verbatim source quote when available")
    transcript_span: tuple[int, int] | None = Field(default=None, description="Character offsets when available")
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SummarySafetyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_diagnosis: Literal[True] = True
    no_treatment_recommendations: Literal[True] = True
    extracted_fields_only: Literal[True] = True
    fail_closed: Literal[True] = True


class SummarySectionId(str, Enum):
    patient_context = "patient_context"
    chief_concern = "chief_concern"
    symptom_summary = "symptom_summary"
    history_context = "history_context"
    medications_allergies = "medications_allergies"
    social_context = "social_context"
    care_goals_or_preferences = "care_goals_or_preferences"
    missing_or_unresolved = "missing_or_unresolved"
    evidence = "evidence"


class SummaryAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    source_fields: list[str] = Field(..., min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class SummarySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: SummarySectionId
    title: str
    atoms: list[SummaryAtom] = Field(default_factory=list)
    omitted_reason: str | None = Field(default=None, description="Why section is empty, e.g. source field absent")

    @model_validator(mode="after")
    def section_has_atoms_or_omission(self):
        if not self.atoms and not self.omitted_reason:
            raise ValueError("SummarySection must contain atoms or an omitted_reason")
        return self


class MissingQuestionSeverity(str, Enum):
    required_for_summary = "required_for_summary"
    helpful_for_review = "helpful_for_review"


class MissingOrUnresolvedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    related_source_fields: list[str] = Field(default_factory=list)
    severity: MissingQuestionSeverity


class ClinicalSummaryStatus(str, Enum):
    draft = "draft"
    blocked = "blocked"


class SummaryValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    field_path: str | None = None
    severity: Literal["error", "warning"]


class ClinicalSummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intake_id: str
    generated_at: datetime
    status: ClinicalSummaryStatus
    safety_policy: SummarySafetyPolicy = Field(default_factory=SummarySafetyPolicy)
    sections: list[SummarySection]
    missing_or_unresolved_questions: list[MissingOrUnresolvedQuestion] = Field(default_factory=list)
    validation_issues: list[SummaryValidationIssue] = Field(default_factory=list)
    source_schema_name: Literal["PatientIntakeExtraction"] = "PatientIntakeExtraction"
    source_schema_version: str

    @model_validator(mode="after")
    def blocked_requires_error(self):
        has_error = any(issue.severity == "error" for issue in self.validation_issues)
        if self.status == ClinicalSummaryStatus.blocked and not has_error:
            raise ValueError("blocked summaries must include at least one error validation issue")
        return self
```

### Fail-closed result envelope

Use an explicit result envelope so API callers never confuse a blocked summary with a valid draft.

```python
class ClinicalSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    summary: ClinicalSummaryDraft | None = None
    errors: list[SummaryValidationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_success_or_errors(self):
        if self.ok and self.summary is None:
            raise ValueError("ok responses must include summary")
        if not self.ok and not self.errors:
            raise ValueError("failed responses must include errors")
        if not self.ok and self.summary is not None and self.summary.status != ClinicalSummaryStatus.blocked:
            raise ValueError("failed responses may only include blocked summaries")
        return self
```

### Optional rendering schema

Keep clinical content separate from display formatting.

```python
class ClinicalSummaryRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["markdown", "json"] = "markdown"
    include_evidence: bool = True
    include_missing_questions: bool = True
```

## 2. Service-Layer Design

### Modules

- `app/schemas/clinical_summary.py`: Pydantic output schemas above.
- `app/services/clinical_summary_engine.py`: Deterministic transformation from `PatientIntakeExtraction` to `ClinicalSummaryResponse`.
- `app/services/summary_validation.py`: Preflight validation and invariant checks.
- `app/services/summary_rendering.py`: Optional markdown rendering from `ClinicalSummaryDraft`; no new clinical facts.

### Pipeline

1. **Preflight validation**
   - Confirm input is a validated `PatientIntakeExtraction` instance, not raw dict unless parsed first by the Sprint 1 schema.
   - Confirm required metadata exists, such as intake id and source schema version.
   - Confirm any field-level consistency rules that must block summary generation.

2. **Source field access through safe getters**
   - Use helper functions that return `(value, field_path, evidence_refs)`.
   - Never read raw transcript text directly in Sprint 2.
   - Never call an LLM for uncontrolled summarization unless the LLM is constrained to a verified set of source atoms and then validated against those atoms. Recommended Sprint 2 implementation should be deterministic first.

3. **Build summary atoms**
   - Convert extracted fields into short factual statements.
   - Every atom must include at least one `source_fields` entry.
   - Include evidence references whenever Sprint 1 extraction provides quote, span, speaker, or confidence.

4. **Build sections**
   - Group atoms into stable sections.
   - Empty sections must carry an explicit `omitted_reason`.
   - Missing or unresolved questions are first-class structured objects, not prose footnotes.

5. **Post-build validation**
   - Enforce no atom without source fields.
   - Enforce no prohibited language patterns suggesting diagnosis or recommendations.
   - Enforce no values absent from the input field allowlist.
   - Return `ok=False` with blocking errors if validation fails.

### Recommended deterministic builder interface

```python
class ClinicalSummaryEngine:
    def build(self, extraction: PatientIntakeExtraction) -> ClinicalSummaryResponse:
        issues = validate_summary_inputs(extraction)
        if any(issue.severity == "error" for issue in issues):
            return blocked_response(extraction, issues)

        draft = ClinicalSummaryDraft(
            intake_id=extraction.intake_id,
            generated_at=clock.utcnow(),
            status=ClinicalSummaryStatus.draft,
            source_schema_version=extraction.schema_version,
            sections=[
                build_patient_context(extraction),
                build_chief_concern(extraction),
                build_symptom_summary(extraction),
                build_history_context(extraction),
                build_medications_allergies(extraction),
                build_missing_or_unresolved_section(extraction),
            ],
            missing_or_unresolved_questions=build_missing_questions(extraction),
            validation_issues=issues,
        )
        output_issues = validate_summary_output(draft, extraction)
        if any(issue.severity == "error" for issue in output_issues):
            draft.status = ClinicalSummaryStatus.blocked
            draft.validation_issues.extend(output_issues)
            return ClinicalSummaryResponse(ok=False, summary=draft, errors=output_issues)
        return ClinicalSummaryResponse(ok=True, summary=draft)
```

## 3. API Endpoint Design

### Endpoint

`POST /api/v1/clinical-summary`

### Request

```json
{
  "intake": { "...": "PatientIntakeExtraction JSON" },
  "render": {
    "format": "json",
    "include_evidence": true,
    "include_missing_questions": true
  }
}
```

### Success response: HTTP 200

```json
{
  "ok": true,
  "summary": {
    "intake_id": "intake_123",
    "generated_at": "2026-07-06T00:00:00Z",
    "status": "draft",
    "safety_policy": {
      "no_diagnosis": true,
      "no_treatment_recommendations": true,
      "extracted_fields_only": true,
      "fail_closed": true
    },
    "sections": [],
    "missing_or_unresolved_questions": [],
    "validation_issues": [],
    "source_schema_name": "PatientIntakeExtraction",
    "source_schema_version": "1.0.0"
  },
  "errors": []
}
```

### Blocked response: HTTP 422

Use HTTP 422 when the submitted extraction is syntactically valid JSON but cannot safely produce a clinical summary.

```json
{
  "ok": false,
  "summary": {
    "status": "blocked",
    "validation_issues": [
      {
        "code": "MISSING_REQUIRED_FIELD",
        "message": "chief_concern is required to generate a doctor-review draft summary",
        "field_path": "chief_concern",
        "severity": "error"
      }
    ]
  },
  "errors": [
    {
      "code": "MISSING_REQUIRED_FIELD",
      "message": "chief_concern is required to generate a doctor-review draft summary",
      "field_path": "chief_concern",
      "severity": "error"
    }
  ]
}
```

### Invalid request response: HTTP 400

Use HTTP 400 for malformed JSON or data that cannot parse as `PatientIntakeExtraction`.

### Idempotency and traceability

- Response should be deterministic for the same input except `generated_at`.
- Tests should inject a frozen clock.
- If the app already has request ids, include request id in logs but not in clinical atoms.

## 4. Validation Invariants

### Input invariants

- Input must parse as `PatientIntakeExtraction`.
- Required identity or tracking fields must be present: `intake_id`, `schema_version`.
- Required clinical anchor fields must be present before summary generation, at minimum chief concern or reason for visit depending on the Sprint 1 schema.
- Any Sprint 1 `extraction_status` or equivalent must indicate successful validated extraction.
- If Sprint 1 marks a field as contradicted or unresolved, Sprint 2 must not choose one side; it must create a missing/unresolved question.
- Evidence references, when present, must point to valid field paths and valid transcript spans.

### Output invariants

- Every `SummaryAtom` has non-empty `source_fields`.
- Every `SummaryAtom.source_fields` path exists in the source extraction.
- Summary text contains no diagnosis language introduced by Sprint 2.
- Summary text contains no treatment recommendation language introduced by Sprint 2.
- Summary text contains no unsupported numbers, dates, medications, allergies, symptoms, or durations.
- Empty sections must have `omitted_reason`.
- Blocked outputs must have `ok=False`, `status=blocked`, and at least one error issue.
- Successful outputs must have `ok=True`, `status=draft`, and no error issues.

### Suggested prohibited phrase checks

These checks are guardrails, not a substitute for source-grounding validation.

- Diagnosis patterns: `consistent with`, `likely`, `suggests`, `diagnosis`, `differential`, `rule out`, `probable`.
- Recommendation patterns: `should start`, `recommend`, `advise`, `prescribe`, `increase dose`, `decrease dose`, `refer to`, `order labs`, `imaging recommended`.

Allow such wording only if it appears verbatim in an extracted source field and is clearly attributed as patient-reported or prior-clinician-reported, not generated as a new conclusion.

## 5. Test Cases

### Unit tests for schemas

1. `SummarySection` rejects empty atoms without `omitted_reason`.
2. `ClinicalSummaryDraft` rejects `blocked` status without error issue.
3. `ClinicalSummaryResponse` rejects `ok=True` without `summary`.
4. `ClinicalSummaryResponse` rejects `ok=False` without errors.
5. `EvidenceRef` rejects confidence outside `[0, 1]`.
6. All summary models reject unknown extra keys.

### Unit tests for engine behavior

1. Minimal valid extraction produces `ok=True` and `status=draft`.
2. Chief concern maps to the chief concern section with source field and evidence reference.
3. Symptom fields produce factual atoms without diagnosis or recommendations.
4. Missing optional fields create missing/unresolved questions or section omissions, not fabricated facts.
5. Contradicted fields create unresolved questions and no selected fact.
6. Missing required anchor field returns HTTP/service-equivalent blocked result.
7. Evidence spans from Sprint 1 are preserved exactly.
8. Renderer output contains only text already present in summary atoms and labels.
9. Frozen clock makes the response snapshot-stable.
10. Same extraction produces identical summary apart from timestamp if clock is not frozen.

### Safety tests

1. Engine rejects generated atom with diagnosis phrase not present in source extraction.
2. Engine rejects generated atom with treatment recommendation phrase not present in source extraction.
3. Engine rejects atom referencing a source field path absent from extraction.
4. Engine rejects unsupported medication, allergy, duration, or date values.
5. Engine never reads raw transcript input in summary generation tests.

### API tests

1. `POST /api/v1/clinical-summary` returns 200 for valid extraction.
2. Endpoint returns 400 for malformed request body.
3. Endpoint returns 422 for parseable extraction missing required summary fields.
4. Response schema matches `ClinicalSummaryResponse`.
5. Evidence references are included by default.
6. Render options can suppress evidence in markdown display without removing evidence from JSON summary.

## 6. File-by-File Implementation Plan

Adjust paths to match the actual Sprint 1 project layout.

1. `app/schemas/clinical_summary.py`
   - Add `EvidenceRef`, `SummarySafetyPolicy`, `SummarySection`, `SummaryAtom`, `MissingOrUnresolvedQuestion`, `SummaryValidationIssue`, `ClinicalSummaryDraft`, and `ClinicalSummaryResponse`.
   - Use `extra="forbid"` everywhere.

2. `app/services/summary_field_access.py`
   - Add safe accessors for Sprint 1 extraction fields.
   - Return normalized field path, value, and evidence references.
   - Centralize support for nested extracted fields.

3. `app/services/summary_validation.py`
   - Add input preflight validation.
   - Add output validation and source-field allowlist checks.
   - Add prohibited generated-language checks.

4. `app/services/clinical_summary_engine.py`
   - Add deterministic section builders.
   - Add `ClinicalSummaryEngine.build()`.
   - Return `ClinicalSummaryResponse`, never raw dict.

5. `app/services/summary_rendering.py`
   - Add markdown renderer for demo and clinician readability.
   - Renderer must not add clinical content beyond labels/headings.

6. `app/api/routes/clinical_summary.py`
   - Add `POST /api/v1/clinical-summary`.
   - Map parse errors to 400 and blocked summaries to 422.

7. `app/main.py` or existing router registration file
   - Register the clinical summary route.

8. `tests/schemas/test_clinical_summary_schema.py`
   - Add schema invariant tests.

9. `tests/services/test_clinical_summary_engine.py`
   - Add deterministic build tests and missing/unresolved behavior tests.

10. `tests/services/test_summary_safety.py`
    - Add diagnosis, treatment, unsupported-field, and source-grounding tests.

11. `tests/api/test_clinical_summary_api.py`
    - Add success, validation, and blocked response tests.

12. `tests/fixtures/patient_intake_extractions.py`
    - Add minimal valid, rich valid, missing required, and contradicted extraction fixtures.

## 7. Streamlit Demo Work for Claude Afterward

Claude should implement the demo only after the backend schemas, service, endpoint, and tests are merged.

### Demo UX additions

- Add a new **Clinical Summary** step after Sprint 1 extraction.
- Display the validated `PatientIntakeExtraction` JSON first, then a **Generate doctor-review draft** button.
- Show summary status prominently: `Draft generated` or `Blocked`.
- Render structured sections in stable order.
- Add an expandable **Evidence** panel for each summary atom.
- Add a persistent **Missing / unresolved questions** panel.
- If blocked, show validation errors and do not show a clinician-style draft summary.

### Demo safety copy

Display a visible notice:

> Draft for clinician review only. This summary does not diagnose, recommend treatment, or add facts beyond the validated intake extraction.

### Demo implementation notes

- Call the backend summary endpoint rather than duplicating summary logic in Streamlit.
- Do not pass raw transcript directly to the summary step.
- Do not use an LLM in the Streamlit layer for rewriting or polishing the summary.
- Preserve and display evidence references from the JSON response.
- Add sample fixtures for: successful draft, blocked missing required field, and unresolved contradiction.

### Demo acceptance criteria

- The demo can generate a summary from a valid extraction fixture.
- The demo shows blocked state for invalid or incomplete extraction fixture.
- Evidence references are visible without expanding raw transcript by default.
- Missing/unresolved questions are visible in both successful and blocked flows.
- No diagnosis or treatment recommendation text is introduced by the UI.
