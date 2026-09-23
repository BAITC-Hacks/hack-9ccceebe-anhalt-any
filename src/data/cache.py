"""Content-checked, immutable JSON weather requests. No credentials in identity."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def response_digest(payload):
    # Exclude server runtime timings, which vary for the same forecast values.
    return digest({k: payload.get(k) for k in ['hourly', 'hourly_units', 'utc_offset_seconds',
                                              'latitude', 'longitude', 'elevation']})


class WeatherCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, identity):
        return self.directory / (digest(identity) + '.json')

    def load(self, identity):
        path = self.path(identity)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text())
            if stored['identity'] != identity or response_digest(stored['payload']) != stored['payload_sha256']:
                raise ValueError('checksum or identity mismatch')
            return stored
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('Weather cache is corrupt; inspect it before retrying') from exc

    def save(self, identity, payload, retrieved_at):
        item = {'identity': identity, 'payload': payload, 'payload_sha256': response_digest(payload),
                'retrieved_at': retrieved_at}
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as stream:
                temp = Path(stream.name)
                json.dump(item, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, self.path(identity))
            except FileExistsError:
                existing = self.load(identity)
                if existing['payload_sha256'] != item['payload_sha256']:
                    raise ValueError('Known weather run changed; refusing to overwrite/backdate its payload')
                return existing
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return item
