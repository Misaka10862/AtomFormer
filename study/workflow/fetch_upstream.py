"""Retrieve the recorded upstream revisions into a new runtime directory."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
PREFIXES = {'libinvent': ('models/', 'running_modes/enums/'),
            'smilesrnn': ('smilesrnn/',), 'promptsmiles': ('promptsmiles/',),
            'drugex': ('drugex/',)}

def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    lock = json.loads((HERE / 'upstream.lock.json').read_text())
    for name, entry in lock.items():
        repository = entry['url'][len('https://github.com/'):]
        url = 'https://codeload.github.com/' + repository + '/tar.gz/' + entry['commit']
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = member.name.split('/', 1)[1]
                if '..' in Path(relative).parts or Path(relative).is_absolute():
                    raise ValueError('Unsafe upstream archive path')
                if relative.startswith(PREFIXES[name]) or relative in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING'):
                    target = destination / name / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
    subprocess.run(['git', 'apply', '-p0', str(HERE / 'libinvent-device.patch')],
                   cwd=destination / 'libinvent', check=True)
    hashes = json.loads((HERE / 'runtime_files.lock.json').read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Upstream runtime checksum mismatch: ' + name)
    print('Verified', len(hashes), 'recorded upstream runtime files.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.destination)
