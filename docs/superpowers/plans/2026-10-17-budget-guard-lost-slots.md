# Budget Guard, Lost Slots Monitoring, and Key Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement minimum pipeline time budget threshold (20s), lost compute slots monitoring in `/health`, and reject placeholder API keys at startup as directed by team lead @AGGIB.

**Architecture:**
- In `backend/app/config.py`: Add validation in `Settings.__post_init__` to fail startup if any key in `api_keys` matches placeholder pattern `change-me-*`.
- In `backend/app/runner.py`: Add minimum budget guard (`min_budget_sec=20.0`), rejecting execution if remaining time before or after acquiring semaphore is less than threshold; track `lost_slots` when watchdog timeout triggers and log `ERROR`.
- In `backend/app/main.py`: Expose `lost_slots` metric in `GET /health`.
- In `backend/tests/`: Add comprehensive tests covering placeholder key rejection, minimum budget failure, and `/health` lost slots tracking.

**Tech Stack:** Python 3.14, FastAPI, Pytest, Pydantic, Threading/Concurrency.

**Spec:** Directives from Agybay Gibatollayev (@AGGIB) in PR #28 review comment #6232535230:
1. Minimum budget threshold: If remaining budget < 20s before start, mark job `failed` with clear error. Do not run pipeline with `time_budget_sec <= 0.001`.
2. Lost slots: Log `ERROR` when thread does not return slot on timeout, expose `lost_slots` in `/health`.
3. Keys: Reject `change-me-*` placeholders at service startup.

## Global Constraints
- Target branch: `feat/backend-budget-guard-lost-slots`.
- All tests must pass (100% pass across full test suite).
- Strict validation: All changes verified by OpenAI Codex CLI with `--dangerously-bypass-approvals-and-sandbox`.

---

### Task 1: Reject Placeholder Keys (`change-me-*`) at Startup
**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Write failing test in `backend/tests/test_config.py`**
  Add `test_placeholder_keys_rejected` verifying that `Settings(api_keys=frozenset({"change-me-jury-key"}), ...)` raises `ValueError`.
- [ ] **Step 2: Run test to verify it fails**
  `pytest backend/tests/test_config.py::test_placeholder_keys_rejected`
- [ ] **Step 3: Implement validation in `Settings.__post_init__`**
  Check `k.startswith("change-me-")` for each key in `self.api_keys`.
- [ ] **Step 4: Run test to verify it passes**
  `pytest backend/tests/test_config.py -v`

---

### Task 2: Minimum Pipeline Budget Guard (20s) in `JobRunner`
**Files:**
- Modify: `backend/app/runner.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing test in `backend/tests/test_api.py`**
  Test `test_insufficient_budget_marks_job_failed_without_calling_pipeline` with job where remaining budget < 20s.
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimum budget threshold in `JobRunner`**
  Reject job with `status="failed"` and clear error if `hard_remaining < min_budget` or `available < min_budget`.
- [ ] **Step 4: Run test to verify it passes**

---

### Task 3: Lost Slots Tracking and `/health` Endpoint Metric
**Files:**
- Modify: `backend/app/runner.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing test in `backend/tests/test_api.py`**
  Test `test_health_reports_lost_slots_on_lingering_thread_timeout`.
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `lost_slots` counter and log `ERROR` in `JobRunner`**
  Log `log.error("job %s: pipeline exceeded hard timeout... retained lingering slot", ...)` and expose `runner.lost_slots`.
- [ ] **Step 4: Update `GET /health` in `backend/app/main.py`**
  Include `"lost_slots": runner.lost_slots` in response dict.
- [ ] **Step 5: Run test to verify it passes**

---

### Task 4: Full Suite Validation and OpenAI Codex Audit
- [ ] **Step 1: Run full `pytest -v` across all backend and ML tests**
- [ ] **Step 2: Run `scripts/load_test.py` against running uvicorn**
- [ ] **Step 3: Run independent audit with OpenAI Codex CLI (`--dangerously-bypass-approvals-and-sandbox`)**
- [ ] **Step 4: Push branch and open Pull Request for Agybay**
