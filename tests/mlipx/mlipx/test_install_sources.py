"""Tests for mlipx.install.sources — source profiles and uv arg builders."""

from __future__ import annotations

import pytest

from mlipx.install.sources import (
    CHINA_SOURCE_CHOICES,
    SOURCE_PROFILES,
    build_offline_args,
    build_package_source_args,
    build_torch_source_args,
    resolve_source,
)


def test_resolve_auto_is_official() -> None:
    assert resolve_source("auto").name == "official"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_source("nope")


def test_official_no_package_override() -> None:
    src = resolve_source("official")
    assert build_package_source_args(src) == []


def test_china_package_uses_tuna() -> None:
    src = resolve_source("china")
    args = build_package_source_args(src)
    assert args == [
        "--index-url",
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    ]


def test_china_torch_uses_aliyun_find_links() -> None:
    src = resolve_source("china")
    args = build_torch_source_args(src, "cu126")
    assert args == [
        "--index-url",
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        "--find-links",
        "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
    ]


@pytest.mark.parametrize(
    ("name", "host"),
    [
        ("china", "mirrors.tuna.tsinghua.edu.cn"),
        ("china-aliyun", "mirrors.aliyun.com/pypi"),
        ("china-ustc", "mirrors.ustc.edu.cn"),
        ("china-tencent", "mirrors.cloud.tencent.com"),
    ],
)
def test_explicit_china_profiles_select_pypi_but_share_torch_mirror(
    name: str, host: str
) -> None:
    src = resolve_source(name)
    assert host in " ".join(build_package_source_args(src))
    torch_args = build_torch_source_args(src, "cpu")
    assert host in " ".join(torch_args)
    assert torch_args[-2:] == [
        "--find-links",
        "https://mirrors.aliyun.com/pytorch-wheels/cpu/",
    ]


def test_official_torch_uses_pytorch_index() -> None:
    src = resolve_source("official")
    args = build_torch_source_args(src, "cu128")
    assert args == ["--index-url", "https://download.pytorch.org/whl/cu128"]


def test_official_cpu_torch_uses_cpu_index() -> None:
    src = resolve_source("official")
    assert build_torch_source_args(src, "cpu") == [
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ]


def test_offline_no_urls() -> None:
    src = resolve_source("offline")
    assert src.offline is True
    assert build_offline_args(src) == ["--offline"]
    assert build_package_source_args(src) == []
    assert build_torch_source_args(src, "cu126") == []
    assert build_torch_source_args(src, "cu128") == []


def test_custom_no_overrides() -> None:
    src = resolve_source("custom")
    assert build_package_source_args(src) == []
    assert build_torch_source_args(src, "cu126") == []
    assert build_torch_source_args(src, "cu128") == []


@pytest.mark.parametrize("name", ["official", "china", "china-aliyun", "offline"])
def test_managed_profile_removes_inherited_resolver_settings(name):
    from mlipx.install.sources import source_environment

    inherited = {
        "UV_INDEX_URL": "https://wrong.example/simple",
        "UV_EXTRA_INDEX_URL": "https://other.example/simple",
        "UV_FIND_LINKS": "https://wrong.example/wheels",
        "UV_INDEX": "named=https://wrong.example/simple",
        "UV_CONSTRAINT": "/tmp/unrelated-constraints.txt",
        "PIP_INDEX_URL": "https://wrong.example/simple",
        "HTTPS_PROXY": "http://proxy.example:8080",
        "SSL_CERT_FILE": "/local/ca.pem",
        "UV_CACHE_DIR": "/local/cache",
        "PATH": "/bin",
    }
    effective = source_environment(resolve_source(name), inherited)
    assert not any("wrong.example" in value for value in effective.values())
    assert "UV_CONSTRAINT" not in effective
    for key in ("HTTPS_PROXY", "SSL_CERT_FILE", "UV_CACHE_DIR", "PATH"):
        assert effective[key] == inherited[key]
    assert "UV_INDEX_URL" in inherited
    assert effective["UV_NO_CONFIG"] == "1"
    if name == "offline":
        assert effective["UV_OFFLINE"] == "1"
        assert effective["UV_PYTHON_DOWNLOADS"] == "never"


def test_custom_source_preserved_and_preview_redacts_credentials():
    from mlipx.install.sources import source_environment, source_environment_summary

    inherited = {
        "UV_INDEX_URL": "https://user:secret@mirror.example/token/private?key=secret",
        "UV_INDEX_PASSWORD": "secret",
    }
    src = resolve_source("custom")
    effective = source_environment(src, inherited)
    assert effective["UV_INDEX_URL"] == inherited["UV_INDEX_URL"]
    preview = "\n".join(source_environment_summary(src, effective))
    assert "mirror.example" in preview
    for secret in ("user:", "secret", "private", "token/"):
        assert secret not in preview


def test_all_profiles_defined() -> None:
    for name in (
        "auto",
        "official",
        "china",
        "china-aliyun",
        "china-ustc",
        "china-tencent",
        "offline",
        "custom",
    ):
        assert name in SOURCE_PROFILES


def test_china_menu_has_stable_numbered_order() -> None:
    assert CHINA_SOURCE_CHOICES == (
        "china",
        "china-aliyun",
        "china-ustc",
        "china-tencent",
        "official",
    )
