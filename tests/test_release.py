"""Credential routing checks use fake tokens and never upload distributions."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release", Path(__file__).resolve().parents[1] / "scripts/release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_pypirc_selects_registry_without_leaking_tokens(tmp_path):
    config = tmp_path / ".pypirc"
    config.write_text(
        "[pypi]\nusername = __token__\npassword = pypi-production-example\n"
        "[testpypi]\nusername = __token__\npassword = pypi-test-example\n"
    )
    token, source = release.publication_token("testpypi", config_path=config, environ={})
    assert token == "pypi-test-example"
    assert token not in source
    token, source = release.publication_token("pypi", config_path=config, environ={})
    assert token == "pypi-production-example"
    assert token not in source


def test_missing_testpypi_does_not_reuse_production_credentials(tmp_path):
    config = tmp_path / ".pypirc"
    config.write_text("[pypi]\nusername = __token__\npassword = pypi-production-example\n")
    with pytest.raises(ValueError, match=r"No \[testpypi\]"):
        release.publication_token(
            "testpypi", config_path=config, environ={"UV_PUBLISH_TOKEN": "pypi-production-example"}
        )


def test_target_environment_overrides_config(tmp_path):
    config = tmp_path / ".pypirc"
    config.write_text("[pypi]\nusername = __token__\npassword = pypi-production-example\n")
    token, source = release.publication_token(
        "testpypi", config_path=config, environ={"TESTPYPI_TOKEN": "pypi-test-example"}
    )
    assert token == "pypi-test-example"
    assert source == "TESTPYPI_TOKEN"


@pytest.mark.parametrize("token", ["account-password", "pypi-invalid\n", "pypi-two words"])
def test_invalid_tokens_are_rejected_without_echoing_values(tmp_path, token):
    with pytest.raises(ValueError) as error:
        release.publication_token(
            "pypi", config_path=tmp_path / "missing", environ={"PYPI_TOKEN": token}
        )
    assert token not in str(error.value)


def test_legacy_environment_without_config(tmp_path):
    token, source = release.publication_token(
        "pypi", config_path=tmp_path / "missing", environ={"UV_PUBLISH_TOKEN": "pypi-example"}
    )
    assert token == "pypi-example"
    assert source == "UV_PUBLISH_TOKEN"


def test_malformed_config_does_not_echo_secrets(tmp_path):
    config = tmp_path / ".pypirc"
    config.write_text("pypi-sensitive-example\n")
    with pytest.raises(ValueError) as error:
        release.publication_token("pypi", config_path=config, environ={})
    assert "pypi-sensitive-example" not in str(error.value)
