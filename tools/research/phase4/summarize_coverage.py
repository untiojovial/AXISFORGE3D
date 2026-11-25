"""Report actual whole-package numerators from the combined pytest coverage JSON."""
import argparse
import json
from pathlib import Path

FIELDS=('covered_lines','num_statements','covered_branches','num_branches')


def counts(items):
    recorded={k:sum(v[k] for v in items) for k in FIELDS}
    for label,numerator,denominator in [('line_percent','covered_lines','num_statements'),('branch_percent','covered_branches','num_branches')]:
        recorded[label]=round(100*recorded[numerator]/recorded[denominator],2) if recorded[denominator] else None
    return recorded


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=json.loads(a.input.read_text())
    if not raw['meta']['branch_coverage']:p.error('branch coverage is required')
    files={name:counts([entry['summary']]) for name,entry in raw['files'].items()}
    report=dict(coverage_version=raw['meta']['version'],measured_by='pytest with subprocess coverage',
                packages={prefix:counts([v for n,v in files.items() if n.startswith(prefix)]) for prefix in
                          ('axisforge_bo/','research_grasping/','research_grasping/phase3/','research_grasping/phase4/')},
                overall=counts(list(files.values())),files=files)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
