import argparse
from multiprocessing import Pool
from pathlib import Path

import torch

from condmolgen.assembly import extract_single_attachment_record, strip_single_attachment_dummy_with_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rooted single-attachment plain-decoration rooted-decoration dataset.")
    parser.add_argument("--input-smi", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def process_smiles(smiles: str):
    record = extract_single_attachment_record(smiles)
    if record is None:
        return None
    rooted = strip_single_attachment_dummy_with_root(record["single_decoration_smiles"])
    if rooted is None:
        return None
    rooted_smiles, anchor_z = rooted
    record["rooted_decoration_smiles"] = rooted_smiles
    record["anchor_z"] = anchor_z
    return record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_smi)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_list = []
    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if args.limit and idx >= args.limit:
                break
            smiles = line.strip()
            if smiles:
                smiles_list.append(smiles)

    records = []
    skipped = 0
    if args.num_workers <= 1:
        iterator = (process_smiles(smiles) for smiles in smiles_list)
        pool = None
    else:
        pool = Pool(processes=args.num_workers)
        iterator = pool.imap(process_smiles, smiles_list)

    try:
        for idx, record in enumerate(iterator, start=1):
            if record is None:
                skipped += 1
            else:
                records.append(record)
            if idx % args.progress_every == 0:
                print(f"processed={idx} kept={len(records)} skipped={skipped}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    torch.save(records, output_path)
    print(f"saved {len(records)} records to {output_path}")
    print(f"skipped {skipped} records")


if __name__ == "__main__":
    main()
