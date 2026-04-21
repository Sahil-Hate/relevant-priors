"""Train final model on full public data + build (cur_desc, prior_desc) -> label lookup table.

Produces:
  app/model.joblib       — final LightGBM classifier
  app/lookup.json        — {"|".join(cur_desc, prior_desc): majority_label} for pairs seen >= MIN_SUPPORT times
  app/feature_names.json — ordered feature names
"""
import json
import numpy as np
import joblib
import lightgbm as lgb
from collections import defaultdict
from features import pair_features, FEATURE_NAMES

PUBLIC_JSON = '/mnt/user-data/uploads/relevant_priors_public.json'
MIN_SUPPORT = 2       # min occurrences to use lookup
CONSISTENCY_THRESH = 0.80  # min majority label fraction to trust lookup

def main():
    with open(PUBLIC_JSON) as f:
        data = json.load(f)
    case_by_id = {c['case_id']: c for c in data['cases']}

    X_rows = []
    y = []
    desc_pair_labels = defaultdict(list)

    for t in data['truth']:
        cid = t['case_id']; sid = t['study_id']
        case = case_by_id.get(cid)
        if not case: continue
        cur = case['current_study']
        prior = next((p for p in case['prior_studies'] if p['study_id']==sid), None)
        if prior is None: continue
        f = pair_features(cur['study_description'], cur['study_date'],
                          prior['study_description'], prior['study_date'])
        X_rows.append([f[k] for k in FEATURE_NAMES])
        label = 1 if t['is_relevant_to_current'] else 0
        y.append(label)
        key = cur['study_description'] + '\x1f' + prior['study_description']
        desc_pair_labels[key].append(label)

    X = np.array(X_rows, dtype=np.float32); y = np.array(y, dtype=np.int32)
    print(f"Training rows: {len(y)}, positive rate: {y.mean():.3f}")

    # Train final model
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, num_leaves=31,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.2,
        random_state=42, verbose=-1,
    )
    model.fit(X, y)
    joblib.dump({'model': model, 'feature_names': FEATURE_NAMES, 'threshold': 0.5}, 'app/model.joblib')

    # Build lookup table
    lookup = {}
    for key, labels in desc_pair_labels.items():
        if len(labels) < MIN_SUPPORT: continue
        mean = sum(labels) / len(labels)
        maj = 1 if mean >= 0.5 else 0
        consistency = max(labels.count(0), labels.count(1)) / len(labels)
        if consistency >= CONSISTENCY_THRESH:
            lookup[key] = {'label': maj, 'support': len(labels), 'consistency': consistency}

    print(f"Lookup entries (MIN_SUPPORT={MIN_SUPPORT}, CONSISTENCY>={CONSISTENCY_THRESH}): {len(lookup)}")
    with open('app/lookup.json', 'w') as f:
        json.dump(lookup, f)

    with open('app/feature_names.json', 'w') as f:
        json.dump(FEATURE_NAMES, f)

    print("Artifacts written to app/")

if __name__ == '__main__':
    main()
