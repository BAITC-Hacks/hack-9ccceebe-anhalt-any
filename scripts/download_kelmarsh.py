"""Download original CC-BY-4.0 Kelmarsh files from Zenodo; verify publisher MD5."""
from pathlib import Path
import hashlib
import time
import urllib.request

FILES = {
    'Kelmarsh_WT_static.csv': 'af3a038f0f7fddfc1608ad0bfc8cf5ba',
    'Kelmarsh_WT_dataSignalMapping.csv': 'ca8fe399ab15ae7111d3214c632efd36',
    'Kelmarsh_SCADA_2017_3083.zip': 'c78263ee52ee0e48e2cb4bbaa1ba211a',
}


def checksum(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1] / 'data/raw/kelmarsh'
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.exists() and checksum(target) == expected:
            print('Already verified:', name, flush=True)
            continue
        url = f'https://zenodo.org/records/16807551/files/{name}?download=1'
        temporary = target.with_suffix(target.suffix + '.part')
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'HackAlem-Energy-ML/0.1'})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if checksum(temporary) != expected:
                    raise ValueError(f'Publisher checksum mismatch: {name}')
                temporary.replace(target)
                print('Downloaded and verified:', name, flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))


if __name__ == '__main__':
    main()
