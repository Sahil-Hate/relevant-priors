"""Start the server in-process and run a test against the public eval JSON."""
import json, sys, time
import requests
import subprocess

def main(public_json: str, base_url: str = 'http://127.0.0.1:8000'):
    with open(public_json) as f:
        data = json.load(f)
    req = {
        'challenge_id': data['challenge_id'],
        'schema_version': data['schema_version'],
        'cases': data['cases'],
    }
    t0 = time.time()
    r = requests.post(f'{base_url}/predict', json=req, timeout=300)
    r.raise_for_status()
    resp = r.json()
    dt = time.time() - t0
    preds = resp['predictions']
    print(f"POST /predict: {len(preds)} predictions in {dt:.2f}s")

    # Score against ground truth
    truth_map = {(t['case_id'], t['study_id']): t['is_relevant_to_current'] for t in data['truth']}
    correct = 0; missing = 0
    for p in preds:
        key = (p['case_id'], p['study_id'])
        if key in truth_map:
            if p['predicted_is_relevant'] == truth_map[key]: correct += 1
        else:
            missing += 1
    total_labels = len(truth_map)
    acc = correct / total_labels
    print(f"Accuracy vs public truth: {correct}/{total_labels} = {acc:.4f}")
    if missing: print(f"  (predictions without a truth match: {missing})")
    return acc

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/relevant_priors_public.json')
