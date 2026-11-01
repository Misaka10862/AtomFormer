"""Connected grouping prevents leakage through any of several identities."""
from collections import defaultdict
import hashlib


def key(seed, text):
    return hashlib.sha256((str(seed) + '\0' + text).encode()).hexdigest()


def leakage_keys(row):
    fields = ('canonical_smiles', 'scaffold_group', 'score_core', 'protein_group', 'pocket_id')
    return [(name, row[name]) for name in fields if row.get(name)]


def grouped_split(records, seed=1729, validation_fraction=0.1, test_fraction=0.1):
    """Official held-out test wins; linked training records are quarantined."""
    rows = [dict(r) for r in records]
    if not rows or validation_fraction <= 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError('Invalid split inputs')
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for identity in leakage_keys(row):
            if identity in seen:
                parent[find(i)] = find(seen[identity])
            else:
                seen[identity] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    official = any(r.get('source_split') == 'test' for r in rows)
    quarantine = []
    output = []
    for indices in groups.values():
        ids = sorted(rows[i]['record_id'] for i in indices)
        group_id = key(seed, '\0'.join(ids))
        x = int(group_id[:16], 16) / 2**64
        locked = any(rows[i].get('source_split') == 'test' for i in indices)
        split = 'test' if locked or (not official and x < test_fraction) else ('validation' if x > 1 - validation_fraction else 'train')
        for i in indices:
            r = dict(rows[i], split=split, split_group=group_id)
            if locked and r.get('source_split') != 'test':
                quarantine.append(dict(r, reason='overlaps_official_test'))
            else:
                output.append(r)
    audit_splits(output)
    return output, quarantine


def audit_splits(records):
    seen, record_ids, counts = {}, set(), defaultdict(int)
    for r in records:
        if r['record_id'] in record_ids:
            raise ValueError('Duplicate record_id')
        record_ids.add(r['record_id'])
        split = r['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Invalid split')
        counts[split] += 1
        for identity in leakage_keys(r):
            if identity in seen and seen[identity] != split:
                raise ValueError('Cross-split overlap: ' + identity[0])
            seen[identity] = split
    return dict(counts)


def stratified_order(records, seed=11):
    """Balanced round-robin order is stable, nested, independent of input order."""
    rows = list(records)
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Policy sampling accepts training only')
    groups = defaultdict(list)
    for r in rows:
        stratum = (len(r['z']) // 5, r.get('attachment_element', r['z'][r['anchor_index']]),
                   r.get('decoration_size', 0) // 5)
        groups[stratum].append(r)
    for group in groups.values():
        group.sort(key=lambda r: key(seed, r['record_id']))
    names = sorted(groups, key=lambda x: key(seed, repr(x)))
    result = []
    for offset in range(max((len(g) for g in groups.values()), default=0)):
        for name in names:
            if offset < len(groups[name]):
                result.append(groups[name][offset])
    return result
