import base64
import json
import shlex
import subprocess
from types import SimpleNamespace

import owner_mlx_transport as adapter
import pytest


def test_remote_atomic_publication_matching_retry_and_conflict(tmp_path):
    path = tmp_path / 'receipt.json'
    data = b'{"status":"pass"}\n'
    sha = adapter.digest(data)
    command = ['python3', '-c', adapter.REMOTE_CODE, 'publish', str(path), sha]
    for _ in range(2):
        result = subprocess.run(command, input=data, capture_output=True)
        assert result.returncode == 0
        assert path.read_bytes() == data
        assert path.with_suffix('.json.sha256').read_bytes() == (sha + '\n').encode()
    other = b'{"status":"different"}'
    result = subprocess.run([*command[:-1], adapter.digest(other)], input=other,
                            capture_output=True)
    assert result.returncode != 0
    assert path.read_bytes() == data


def test_publish_readback_and_local_digest_fail_closed(tmp_path):
    local = tmp_path / 'local.json'
    data = b'{"status":"pass"}\n'
    local.write_bytes(data)
    sha = adapter.digest(data)
    calls = []

    def mock(command, **kwargs):
        calls.append((shlex.split(command), kwargs))
        assert kwargs['timeout'] == 45 and kwargs['capture_output']
        if calls[-1][0][3] == 'publish':
            assert kwargs['input'] == data
            return SimpleNamespace(returncode=0, stdout=b'')
        packet = {'receipt': base64.b64encode(data).decode(),
                  'checksum': base64.b64encode((sha + '\n').encode()).decode()}
        return SimpleNamespace(returncode=0, stdout=json.dumps(packet).encode())

    result = adapter.publish_receipt(mock, '/content/receipt.json', local, sha)
    assert result[0] == result[1] == data
    assert result[3]['sha256'] == sha
    assert len(calls) == 2
    with pytest.raises(ValueError, match='hash mismatch'):
        adapter.publish_receipt(mock, '/content/receipt.json', local, '0' * 64)
    assert len(calls) == 2


def test_manifest_exact_bytes_partial_network_and_bad_readback(tmp_path):
    data = b'{ "seeds": [] }\n'
    def mock(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=data)
    assert adapter.read_manifest(mock, '/content/manifest.json') == data
    with pytest.raises(json.JSONDecodeError):
        adapter.read_manifest(lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'{'),
                              '/content/manifest.json')
    with pytest.raises(RuntimeError, match='request failed'):
        adapter.read_manifest(lambda *a, **k: SimpleNamespace(returncode=1, stdout=b''),
                              '/content/manifest.json')
    local = tmp_path / 'receipt.json'
    local.write_bytes(data)
    wrong = {'receipt': base64.b64encode(data).decode(),
             'checksum': base64.b64encode(b'wrong').decode()}
    with pytest.raises(ValueError, match='readback mismatch'):
        adapter.publish_receipt(
            lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(wrong).encode()),
            '/content/receipt.json', local, adapter.digest(data))


@pytest.mark.parametrize('path', ['/etc/passwd', '/content/../etc/passwd',
                                  '/content//receipt.json', '/content'])
def test_remote_path_bounds(path):
    with pytest.raises(ValueError):
        adapter.remote_path(path)
