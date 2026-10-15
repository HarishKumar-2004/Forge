import sys

import pytest

from tests.e2e.fake_ollama_server import FakeOllamaServer
from tests.e2e.pty_app import PtyApp, make_app_dir

if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items):
    for item in items:
        if "/e2e/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture
def server():
    srv = FakeOllamaServer().start()
    yield srv
    srv.stop()


@pytest.fixture
def app_dir(tmp_path):
    return make_app_dir(tmp_path)


@pytest.fixture
def launch(app_dir, server):
    """launch(*args) -> a running PtyApp talking to the fake server; always killed at teardown."""
    apps = []

    def _launch(*args, host=None, **kw):
        app = PtyApp(app_dir, list(args), host or server.host, **kw)
        apps.append(app)
        return app

    yield _launch
    for app in apps:
        app.close()
