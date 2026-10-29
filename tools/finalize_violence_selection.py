"""Record the predeclared selection gate without running holdout inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.runtime/violence-improvement-20261028'


def main():
    rows = []
    sources = {}
    for model in ('x3d', 'r2plus1d'):
        path = WORK / ('selection-' + model) / 'report.json'
        raw = path.read_bytes()
        report = json.loads(raw)
        if not report['complete'] or len(report['cases']) != 36:
            raise ValueError('Incomplete selection report: ' + model)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for mode, counts in report['summary'].items():
            if counts['TP'] + counts['FN'] != 18 or counts['FP'] + counts['TN'] != 18:
                raise ValueError('Unexpected selection class totals')
            sensitivity = counts['TP'] / 18
            specificity = counts['TN'] / 18
            rows.append(dict(model=model, mode=mode, **counts,
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity+specificity)/2,
                gate_passed=counts['FP'] <= 1))
    eligible = sorted((r for r in rows if r['gate_passed']),
                      key=lambda r: r['balanced_accuracy'], reverse=True)
    result = {'gate': 'At most 1 false positive among 18 selection negatives',
        'results': rows, 'gate_passed': bool(eligible),
        'selected_model': eligible[0]['model'] if eligible else None,
        'selected_mode': eligible[0]['mode'] if eligible else None,
        'validation_executed': False, 'cross_source_inference_executed': False,
        'source_report_sha256': sources,
        'decision': 'Eligible for holdout evaluation only' if eligible else
                    'No promotion; preserve holdout, causal cross-source testing deferred',
        'live_enabled': False}
    target = WORK / 'SELECTION-DECISION.json'
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError('Prior decision differs; preserve it')
    else:
        target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
