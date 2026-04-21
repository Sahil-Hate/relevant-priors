"""FastAPI inference server for the 'relevant priors' challenge.

POST /predict  — accepts cases, returns predictions.
GET  /health   — health check.
GET  /         — info.

Predict logic:
  1. Exact-description match (normalized) -> True.
  2. Lookup table (cur_desc, prior_desc) built from public training -> memorized label.
  3. Otherwise: LightGBM model over engineered features.
Per-case featurization is batched; no external API calls.
"""
import json
import os
import time
import logging
import uuid
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from features import pair_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('priors')

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, 'model.joblib'))
MODEL = BUNDLE['model']
THRESHOLD = float(BUNDLE.get('threshold', 0.5))

with open(os.path.join(HERE, 'lookup.json')) as f:
    LOOKUP = json.load(f)
log.info(f"Loaded model (threshold={THRESHOLD}) and lookup ({len(LOOKUP)} entries)")

# Caches (process-local)
_feat_cache: dict = {}   # (cur_desc, cur_date, prior_desc, prior_date) -> list[float]
_prob_cache: dict = {}   # same key -> float prob

def _norm(s: str) -> str:
    return ' '.join((s or '').lower().split())

# ---------- Request / response schemas ----------

class Study(BaseModel):
    study_id: str
    study_description: str = ''
    study_date: Optional[str] = None

class Case(BaseModel):
    case_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    current_study: Study
    prior_studies: List[Study] = Field(default_factory=list)

class PredictRequest(BaseModel):
    challenge_id: Optional[str] = None
    schema_version: Optional[int] = None
    generated_at: Optional[str] = None
    cases: List[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class PredictResponse(BaseModel):
    predictions: List[Prediction]

# ---------- App ----------

app = FastAPI(title='Relevant Priors', version='1.0.0')

@app.get('/')
def root():
    return {'service': 'relevant-priors', 'status': 'ok',
            'endpoints': ['/predict (POST)', '/health (GET)'],
            'lookup_entries': len(LOOKUP), 'threshold': THRESHOLD}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    rid = request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    t0 = time.time()
    n_cases = len(req.cases)
    n_priors = sum(len(c.prior_studies) for c in req.cases)
    log.info(f"[{rid}] predict: cases={n_cases} priors={n_priors}")

    predictions: List[Prediction] = []

    # Collect pairs needing model inference into a batch; apply exact/lookup first
    batch_keys = []       # list of (case_id, study_id, cache_key)
    batch_feats = []      # list of feature vectors
    batch_skip = []       # for (case_id, study_id, bool) already decided

    for case in req.cases:
        cur = case.current_study
        cur_desc = cur.study_description or ''
        cur_date = cur.study_date or ''
        cur_norm = _norm(cur_desc)

        for prior in case.prior_studies:
            pr_desc = prior.study_description or ''
            pr_date = prior.study_date or ''

            # 1) Exact match on normalized descriptions
            if cur_norm and cur_norm == _norm(pr_desc):
                batch_skip.append((case.case_id, prior.study_id, True))
                continue

            # 2) Lookup table from training descriptions
            key_str = cur_desc + '\x1f' + pr_desc
            if key_str in LOOKUP:
                batch_skip.append((case.case_id, prior.study_id, bool(LOOKUP[key_str]['label'])))
                continue

            # 3) Model path — gather features, with caching on (cur_desc, cur_date, pr_desc, pr_date)
            ckey = (cur_desc, cur_date, pr_desc, pr_date)
            if ckey in _feat_cache:
                feats = _feat_cache[ckey]
            else:
                f = pair_features(cur_desc, cur_date, pr_desc, pr_date)
                feats = [f[k] for k in FEATURE_NAMES]
                _feat_cache[ckey] = feats

            batch_keys.append((case.case_id, prior.study_id, ckey))
            batch_feats.append(feats)

    # Model inference (batched)
    if batch_feats:
        # Separate already-cached probs
        uncached_idx = [i for i, (_, _, k) in enumerate(batch_keys) if k not in _prob_cache]
        if uncached_idx:
            X = np.array([batch_feats[i] for i in uncached_idx], dtype=np.float32)
            probs = MODEL.predict_proba(X)[:, 1]
            for j, i in enumerate(uncached_idx):
                _prob_cache[batch_keys[i][2]] = float(probs[j])
        for cid, sid, k in batch_keys:
            p = _prob_cache[k]
            predictions.append(Prediction(case_id=cid, study_id=sid,
                                          predicted_is_relevant=bool(p >= THRESHOLD)))

    # Decided-upfront cases
    for cid, sid, is_rel in batch_skip:
        predictions.append(Prediction(case_id=cid, study_id=sid, predicted_is_relevant=is_rel))

    dt = time.time() - t0
    log.info(f"[{rid}] done in {dt:.2f}s  out={len(predictions)}  "
             f"(exact/lookup={len(batch_skip)} model={len(batch_keys)})")
    return PredictResponse(predictions=predictions)
