"""Feature engineering for (current_study, prior_study) pair relevance prediction (v2)."""
import re
import math
from datetime import datetime

MODALITY_PATTERNS = [
    ('ct',     [r'\bct\b', r'\bcat scan\b']),
    ('mri',    [r'\bmri\b', r'\bmr\b']),
    ('us',     [r'\bus\b', r'\bultrasound\b', r'\bultrasnd\b', r'\bendovag', r'\bdoppler\b', r'\bsono']),
    ('xr',     [r'\bxr\b', r'\bx-?ray\b', r'\bradiograph']),
    ('mam',    [r'\bmam\b', r'\bmamm', r'\btomo\b']),
    ('nm',     [r'\bnm\b', r'\bnuclear\b', r'\bspect\b', r'\bbone scan\b', r'\bscintig']),
    ('pet',    [r'\bpet\b']),
    ('echo',   [r'\becho\b', r'\bechocardi', r'\btte\b', r'\btee\b']),
    ('fluoro', [r'\bfluoro', r'\bloopogram\b', r'\besophagram\b', r'\bswallow\b', r'\bugi\b', r'\benema\b']),
    ('angio',  [r'\bangio', r'\barteriog']),
    ('bmd',    [r'\bbmd\b', r'\bdexa\b', r'\bbone densit']),
    ('eeg',    [r'\beeg\b']),  # neurology signal — paired with head/brain imaging
    ('bx',     [r'\bbiopsy\b', r'\bfna\b', r'\bbx\b']),
]

# Body regions — spine segments kept SEPARATE (major error source in v1)
REGION_PATTERNS = [
    ('breast',    [r'\bbreast', r'\bmam', r'\btomo\b']),
    ('chest',     [r'\bchest\b', r'\bthorax\b', r'\blung', r'\bribs?\b', r'\bsternum\b']),
    ('abdomen',   [r'\babd\b', r'\babdom', r'\bliver\b', r'\bhepatic\b', r'\bkidney', r'\brenal\b', r'\bbladder\b',
                   r'\bpancreas', r'\bspleen', r'\bgallbladder\b', r'\bgb\b', r'\bbiliary\b', r'\benterog']),
    ('pelvis',    [r'\bpelv', r'\bhip\b', r'\bhips\b', r'\bpelvic\b']),
    ('head',      [r'\bhead\b', r'\bbrain\b', r'\bskull\b', r'\bcranium\b', r'\beeg\b', r'\biac\b']),
    ('neck',      [r'\bneck\b', r'\bthyroid\b', r'\bcarotid\b', r'\bsoft tissue neck\b']),
    ('cspine',    [r'\bc-?spine\b', r'\bcervical spine\b', r'\bcervicl\b', r'\bcervical\b(?! cancer)']),
    ('tspine',    [r'\bt-?spine\b', r'\bthoracic spine\b']),
    ('lspine',    [r'\bl-?spine\b', r'\blumbar\b']),
    ('sspine',    [r'\bs-?spine\b', r'\bsacral\b', r'\bsacrum\b', r'\bcoccyx\b']),
    ('spine_oth', [r'\bscoliosis\b']),
    ('shoulder',  [r'\bshoulder\b', r'\bclavicle\b', r'\bscapula\b']),
    ('elbow',     [r'\belbow\b']),
    ('wrist',     [r'\bwrist\b', r'\bhand\b', r'\bfinger']),
    ('upper_ext', [r'\bhumerus\b', r'\bforearm\b', r'\bue\b', r'\bupper ext', r'\bupper rgt extrem\b', r'\bupper extrem']),
    ('lower_ext', [r'\bfemur\b', r'\btibia\b', r'\bfibula\b', r'\ble\b(?! contrast)', r'\blower ext', r'\blower extrem']),
    ('knee',      [r'\bknee\b']),
    ('ankle',     [r'\bankle\b', r'\bankler\b', r'\bfoot\b', r'\btoe']),
    ('heart',     [r'\bheart\b', r'\bcardi', r'\bmyocard', r'\bcoronary\b', r'\becho\b', r'\btte\b', r'\btee\b']),
    ('face',      [r'\bfacial\b', r'\bmaxillo', r'\bsinus', r'\bmandib', r'\borbit', r'\btm ?j\b']),
    ('vascular',  [r'\bvas\b', r'\bvenous\b', r'\barterial\b', r'\bvein', r'\bartery\b', r'\bdvt\b', r'\baorta\b', r'\bangio']),
    ('gu',        [r'\bgu\b', r'\burethra', r'\bprostate\b', r'\bscrotum\b', r'\btestic', r'\buterus\b', r'\bovar']),
    ('bone',      [r'\bbone\b(?! densit)']),
    ('pediatric', [r'\bpediatric\b', r'\binfant\b']),
]

# Related regions — presence in same group gives a "related" bonus (exact=0).
RELATED_REGIONS = [
    {'chest', 'heart'},
    {'breast', 'chest'},
    {'abdomen', 'pelvis'},
    {'pelvis', 'gu'},
    {'head', 'neck'},
    {'head', 'face'},
    {'neck', 'cspine'},
    {'abdomen', 'lspine'},          # retroperitoneal proximity
    {'chest', 'tspine'},
    # spine segments: adjacent ones ARE sometimes co-read, but less than same-segment
    {'cspine', 'tspine'},
    {'tspine', 'lspine'},
    {'lspine', 'sspine'},
    {'upper_ext', 'shoulder', 'elbow', 'wrist'},
    {'lower_ext', 'knee', 'ankle'},
    {'chest', 'vascular'},          # CT angio chest, etc.
    {'abdomen', 'vascular'},
]

def _norm(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', (s or '').lower())

def _detect(patterns, text):
    hits = set()
    for name, pats in patterns:
        for pat in pats:
            if re.search(pat, text):
                hits.add(name)
                break
    return hits

STOPWORDS = {'', 'and', 'or', 'the', 'with', 'without', 'of', 'a', 'an', 'to', 'for', 'by'}

def char_ngrams(s, n=3):
    s = f" {s.strip()} "
    return set(s[i:i+n] for i in range(len(s)-n+1))

def parse_study(desc: str):
    t = _norm(desc)
    modalities = _detect(MODALITY_PATTERNS, t)
    regions = _detect(REGION_PATTERNS, t)

    with_contrast = bool(re.search(r'\bwith contrast\b|\bw con\b|\bw/con\b|\bwith con\b|\bw contrast\b|\bw/c\b|\bwith cntrst\b|\bw cntrst\b', t)) \
                    and not re.search(r'\bwithout contrast\b|\bwo contrast\b|\bw/o con\b|\bwo con\b|\bwithout cntrst\b', t)
    without_contrast = bool(re.search(r'\bwithout contrast\b|\bwo contrast\b|\bwo con\b|\bwithout cntrst\b|\bwo cntrst\b|\bwo/\b|\bw/o\b|\bno contrast\b', t))

    left = bool(re.search(r'\bleft\b|\blt\b|, l$', t))
    right = bool(re.search(r'\bright\b|\brt\b|, r$', t))
    bilateral = bool(re.search(r'\bbilateral\b|\bbilat\b|\bboth\b', t))
    screening = bool(re.search(r'\bscreen', t))
    diagnostic = bool(re.search(r'\bdiagn|\bdx\b', t))
    biopsy = bool(re.search(r'\bbiopsy\b|\bfna\b|\bbx\b|\baspirat', t))
    limited = bool(re.search(r'\blimited\b', t))

    tokens = set(t.split()) - STOPWORDS
    return {
        'desc': desc, 'norm': t, 'tokens': tokens,
        'modalities': modalities, 'regions': regions,
        'with_contrast': with_contrast, 'without_contrast': without_contrast,
        'left': left, 'right': right, 'bilateral': bilateral,
        'screening': screening, 'diagnostic': diagnostic,
        'biopsy': biopsy, 'limited': limited,
    }

def region_match_score(a_regions, b_regions):
    if not a_regions or not b_regions:
        return 0.0, 0.0
    exact = 1.0 if a_regions & b_regions else 0.0
    related = 0.0
    if not exact:
        for group in RELATED_REGIONS:
            if a_regions & group and b_regions & group:
                related = 1.0
                break
    return exact, related

def modality_match_score(a_mods, b_mods):
    if not a_mods or not b_mods:
        return 0.0, 0.0
    exact = 1.0 if a_mods & b_mods else 0.0
    groupings = [
        {'ct', 'mri'}, {'ct', 'xr'}, {'mri', 'xr'},
        {'mam', 'us'}, {'mam', 'mri'}, {'mam', 'bx'},
        {'nm', 'pet'}, {'nm', 'ct'}, {'pet', 'ct'},
        {'us', 'ct'}, {'us', 'mri'}, {'us', 'bx'},
        {'echo', 'nm'}, {'echo', 'ct'}, {'echo', 'mri'},
        {'eeg', 'ct'}, {'eeg', 'mri'},  # neuro pairing
        {'ct', 'angio'}, {'mri', 'angio'},
    ]
    related = 0.0
    if not exact:
        for g in groupings:
            if a_mods & g and b_mods & g:
                related = 1.0
                break
    return exact, related

def days_between(date_a: str, date_b: str):
    try:
        da = datetime.fromisoformat(date_a)
        db = datetime.fromisoformat(date_b)
        return abs((da - db).days)
    except Exception:
        return -1

def pair_features(current_desc, current_date, prior_desc, prior_date):
    a = parse_study(current_desc)
    b = parse_study(prior_desc)

    exact_desc = 1.0 if a['norm'] == b['norm'] else 0.0

    tok_inter = a['tokens'] & b['tokens']
    tok_union = a['tokens'] | b['tokens']
    jaccard = len(tok_inter) / len(tok_union) if tok_union else 0.0
    overlap = len(tok_inter) / min(len(a['tokens']), len(b['tokens'])) if a['tokens'] and b['tokens'] else 0.0

    # Character n-gram similarity (catches abbrev variants)
    a_ng = char_ngrams(a['norm']); b_ng = char_ngrams(b['norm'])
    ng_inter = a_ng & b_ng; ng_union = a_ng | b_ng
    char_jaccard = len(ng_inter) / len(ng_union) if ng_union else 0.0

    region_exact, region_rel = region_match_score(a['regions'], b['regions'])
    mod_exact, mod_rel = modality_match_score(a['modalities'], b['modalities'])

    # Spine-segment-specific features: if both are spine but different segments, PENALIZE
    spine_segs = {'cspine', 'tspine', 'lspine', 'sspine'}
    a_spine = a['regions'] & spine_segs
    b_spine = b['regions'] & spine_segs
    both_spine = 1.0 if a_spine and b_spine else 0.0
    spine_same_seg = 1.0 if a_spine and b_spine and (a_spine & b_spine) else 0.0
    spine_diff_seg = 1.0 if both_spine and not (a_spine & b_spine) else 0.0

    dd = days_between(current_date, prior_date)
    days_log = math.log1p(dd) if dd >= 0 else -1.0
    days_missing = 1.0 if dd < 0 else 0.0
    days_5y_plus = 1.0 if dd > 365*5 else 0.0
    days_1y_minus = 1.0 if 0 <= dd < 365 else 0.0

    same_laterality = 1.0 if (a['left'] and b['left']) or (a['right'] and b['right']) or (a['bilateral'] and b['bilateral']) else 0.0
    opposite_laterality = 1.0 if (a['left'] and b['right']) or (a['right'] and b['left']) else 0.0
    contrast_match = 1.0 if (a['with_contrast'] == b['with_contrast']) and (a['without_contrast'] == b['without_contrast']) else 0.0

    return {
        'exact_desc': exact_desc,
        'jaccard': jaccard,
        'overlap_coef': overlap,
        'char_jaccard': char_jaccard,
        'inter_tokens': float(len(tok_inter)),
        'a_n_tokens': float(len(a['tokens'])),
        'b_n_tokens': float(len(b['tokens'])),
        'region_exact': region_exact,
        'region_related': region_rel,
        'region_any_a': 1.0 if a['regions'] else 0.0,
        'region_any_b': 1.0 if b['regions'] else 0.0,
        'mod_exact': mod_exact,
        'mod_related': mod_rel,
        'mod_any_a': 1.0 if a['modalities'] else 0.0,
        'mod_any_b': 1.0 if b['modalities'] else 0.0,
        'both_spine': both_spine,
        'spine_same_seg': spine_same_seg,
        'spine_diff_seg': spine_diff_seg,
        'days_log': days_log,
        'days_missing': days_missing,
        'days_5y_plus': days_5y_plus,
        'days_1y_minus': days_1y_minus,
        'same_laterality': same_laterality,
        'opposite_laterality': opposite_laterality,
        'contrast_match': contrast_match,
        'a_screening': 1.0 if a['screening'] else 0.0,
        'b_screening': 1.0 if b['screening'] else 0.0,
        'a_diagnostic': 1.0 if a['diagnostic'] else 0.0,
        'b_diagnostic': 1.0 if b['diagnostic'] else 0.0,
        'a_biopsy': 1.0 if a['biopsy'] else 0.0,
        'b_biopsy': 1.0 if b['biopsy'] else 0.0,
    }

FEATURE_NAMES = list(pair_features(
    "CT CHEST WITH CONTRAST", "2026-01-01", "XR chest 2V", "2024-01-01"
).keys())
