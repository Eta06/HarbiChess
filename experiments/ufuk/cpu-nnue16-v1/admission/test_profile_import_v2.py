import subprocess
import sys
from pathlib import Path


def test_fresh_interpreter_cprofile_resolves_stdlib_from_shadow_directory():
    directory = Path(__file__).parent
    script = (
        "import teacher_profile_v2 as helper; "
        "helper.isolate_stdlib_profile(); "
        "import cProfile, profile; "
        "assert hasattr(profile, 'run') and hasattr(profile, '_Utils'); "
        "assert str(helper.__file__) != str(profile.__file__); "
        "print('PASS-stdlib-profile-no-Torch-no-forward')"
    )
    p = subprocess.run([sys.executable, "-c", script], cwd=directory,
                       capture_output=True, text=True, timeout=10)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "PASS-stdlib-profile-no-Torch-no-forward"
