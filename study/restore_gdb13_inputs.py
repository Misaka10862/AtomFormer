"""Restore fixed GDB13 inputs from an upstream file and study selectors.

Obtain the upstream file under the provider's terms. This script does not
download or redistribute the upstream structures and never creates a new split.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from condmolgen.assembly import extract_single_attachment_record, strip_single_attachment_dummy_with_root


def restore_record(source_smiles, selector):
    if hashlib.sha256(source_smiles.encode()).hexdigest() != selector['source_smiles_sha256']:
        raise ValueError('Source molecule identity differs from the archived selector')
    record = extract_single_attachment_record(source_smiles, reference_geometry=selector['numerical_fields'])
    if record is None:
        raise ValueError('The archived molecule could not be reconstructed with this RDKit version')
    target = strip_single_attachment_dummy_with_root(record['single_decoration_smiles'])
    if target is None:
        raise ValueError('Attachment-first target reconstruction failed')
    record['rooted_decoration_smiles'], record['anchor_z'] = target
    for key, digest in selector['string_sha256'].items():
        if hashlib.sha256(record[key].encode()).hexdigest() != digest:
            raise ValueError('Reconstructed field differs: ' + key + '. Match the archived RDKit version.')
    for key, value in selector['numerical_fields'].items():
        if record[key] != value:
            raise ValueError('Reconstructed numerical field differs: ' + key)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-gz', type=Path, required=True)
    parser.add_argument('--selectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.selectors / 'manifest.json').read_text())
    with gzip.open(args.upstream_gz, 'rt') as f:
        smiles = [line.split()[0] for line in f if line.strip()]
    raw = ('\n'.join(smiles) + '\n').encode()
    if len(smiles) != manifest['working_smiles_lines'] or hashlib.sha256(raw).hexdigest() != manifest['working_smiles_sha256']:
        raise ValueError('Wrong upstream GDB13 file or first-column extraction')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    for split in manifest['splits']:
        records = []
        with gzip.open(args.selectors / split['file'], 'rt') as f:
            for line in f:
                selector = json.loads(line)
                if selector['record_index'] != len(records):
                    raise ValueError('Selector ordering is not contiguous')
                records.append(restore_record(smiles[selector['source_line_index']], selector))
        if len(records) != split['records']:
            raise ValueError('Incomplete fixed split')
        torch.save(records, args.output / (split['split'] + '.pt'))
        print(split['split'], len(records), 'records, all archived field hashes matched')


if __name__ == '__main__':
    main()
