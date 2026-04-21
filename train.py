"""Train a LightGBM classifier on labeled (current, prior) pairs using case-level CV."""
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from features import pair_features, FEATURE_NAMES

PUBLIC_JSON = '/mnt/user-data/uploads/relevant_priors_public.json'

def load_pairs(path):
    with open(path) as f:
        data = json.load(f)
    case_by_id = {c['case_id']: c for c in data['cases']}
    rows = []
    for t in data['truth']:
        cid = t['case_id']
        sid = t['study_id']
        case = case_by_id.get(cid)
        if not case:
            continue
        cur = case['current_study']
        prior = None
        for p in case['prior_studies']:
            if p['study_id'] == sid:
                prior = p
                break
        if prior is None:
            continue
        feats = pair_features(cur['study_description'], cur['study_date'],
                              prior['study_description'], prior['study_date'])
        rows.append({
            'case_id': cid,
            'study_id': sid,
            'features': feats,
            'label': 1 if t['is_relevant_to_current'] else 0,
        })
    return rows

def rows_to_matrix(rows):
    X = np.array([[r['features'][k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    y = np.array([r['label'] for r in rows], dtype=np.int32)
    groups = np.array([r['case_id'] for r in rows])
    return X, y, groups

if __name__ == '__main__':
    print("Loading and featurizing...")
    rows = load_pairs(PUBLIC_JSON)
    print(f"Rows: {len(rows)}")
    X, y, groups = rows_to_matrix(rows)
    print(f"X shape: {X.shape}, positive rate: {y.mean():.3f}")

    # 5-fold CV grouped by case_id
    gkf = GroupKFold(n_splits=5)
    oof_prob = np.zeros(len(y), dtype=np.float32)
    fold_accs = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(Xtr, ytr,
                  eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        prob = model.predict_proba(Xva)[:, 1]
        oof_prob[va] = prob
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(yva, pred)
        auc = roc_auc_score(yva, prob)
        fold_accs.append(acc)
        print(f"Fold {fold}: n_va={len(va):5d}  acc={acc:.4f}  auc={auc:.4f}")

    # Overall OOF metrics
    oof_pred = (oof_prob >= 0.5).astype(int)
    print(f"\nOOF accuracy @ 0.5:   {accuracy_score(y, oof_pred):.4f}")
    print(f"OOF AUC:             {roc_auc_score(y, oof_prob):.4f}")

    # Threshold sweep
    print("\nThreshold sweep:")
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = (oof_prob >= th).astype(int)
        print(f"  th={th:.2f}  acc={accuracy_score(y, pred):.4f}  "
              f"precision_of_true={(y[pred==1]==1).mean() if (pred==1).any() else 0:.3f}  "
              f"recall_of_true={(pred[y==1]==1).mean():.3f}")

    # Train final model on everything
    print("\nTraining final model on full data...")
    final = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    final.fit(X, y)
    import joblib
    joblib.dump({'model': final, 'feature_names': FEATURE_NAMES, 'threshold': 0.5}, 'model.joblib')
    print("Saved model.joblib")

    # Feature importance
    print("\nFeature importance (gain):")
    imps = sorted(zip(FEATURE_NAMES, final.booster_.feature_importance(importance_type='gain')), key=lambda x: -x[1])
    for name, imp in imps:
        print(f"  {name:25s}  {imp:10.1f}")
