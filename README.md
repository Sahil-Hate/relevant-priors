# Relevant Priors API

Classifier that decides, for each prior examination in a radiology case, whether
it should be shown to the radiologist reading the current examination.

## Approach (short version)

Three-tier decision pipeline:

1. **Exact description match** — if the normalized current and prior
   descriptions are identical, return `True`. On the public split, exact-matched
   pairs are `True` 99.9% of the time, so this is free accuracy.
2. **Memorized lookup** — a `(current_description, prior_description) → label`
   table built from the public training set. Only stores pairs with ≥2
   occurrences and ≥80% label consistency. Covers ~4,800 distinct pairs.
3. **LightGBM classifier** over 31 engineered features (modality, body region
   with separate spine segments, laterality, contrast, dates, token + char
   n-gram similarity). Applied to everything else.

Cross-validated accuracy on the public split: **94.3%** (5-fold grouped by
`case_id`, honest out-of-sample). On the full public split end-to-end (with
lookup benefiting from training memory), the pipeline hits 97.7%.

No LLM calls. No external services. Pure in-process inference. ~2.5 seconds for
996 cases / 27,614 priors on a single CPU.

## Files

```
server.py           — FastAPI app: POST /predict, GET /health, GET /
features.py         — Feature engineering (31 features per pair)
model.joblib        — Trained LightGBM classifier + threshold (0.55)
lookup.json         — Memorized (cur_desc, prior_desc) → label entries
feature_names.json  — Ordered feature names (reference)
requirements.txt    — Python dependencies
Dockerfile          — Container build for deployment
render.yaml         — Render.com service config (free tier, Docker)
test_server.py      — Smoke test against the public eval JSON
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @relevant_priors_public.json
```

## Deploying (Render)

```bash
# From this directory
git init && git add . && git commit -m "initial"
# Push to GitHub, then connect repo to Render as a Docker web service.
# render.yaml will be picked up automatically.
```

Health check endpoint: `/health` (returns 200 with `{"status":"ok"}`).

## API

### Request

```json
{
  "challenge_id": "relevant-priors-v1",
  "schema_version": 1,
  "cases": [
    {
      "case_id": "...",
      "patient_id": "...",
      "patient_name": "...",
      "current_study": {
        "study_id": "...",
        "study_description": "...",
        "study_date": "YYYY-MM-DD"
      },
      "prior_studies": [
        { "study_id": "...", "study_description": "...", "study_date": "YYYY-MM-DD" }
      ]
    }
  ]
}
```

### Response

```json
{
  "predictions": [
    { "case_id": "...", "study_id": "...", "predicted_is_relevant": true }
  ]
}
```

One prediction is returned for every prior study in the request, keyed by
`(case_id, study_id)`.

## Performance notes

- Model file: ~1.7 MB
- Lookup file: ~500 KB
- Featurization throughput: ~4,700 pairs/sec single-threaded
- Model inference: 27k pairs in ~0.4s (LightGBM native batch predict)
- Process-local caches on featurization and model output keyed by
  `(current_desc, current_date, prior_desc, prior_date)` so retries and
  repeated pairs within a session are free.

## Building artifacts from scratch

If you want to retrain:

```bash
# From repo root (one level above app/)
python3 train.py            # 5-fold CV + save model.joblib at root
python3 build_artifacts.py  # Produces app/model.joblib + app/lookup.json
```
