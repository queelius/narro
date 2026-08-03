from pathlib import Path

import pytest

from muse.federation.nodes import NodeSpec, load_nodes


def test_cli_nodes_plain_and_named():
    nodes = load_nodes(cli_nodes=["http://a:8000/", "b=http://b:8000"])
    assert nodes[0] == NodeSpec(url="http://a:8000", name="a", token=None)  # trailing slash stripped, name from host
    assert nodes[1] == NodeSpec(url="http://b:8000", name="b", token=None)


def test_yaml_nodes_and_dedup(tmp_path):
    p = tmp_path / "federation.yaml"
    p.write_text("nodes:\n  - {url: http://a:8000, name: alpha, token: t1}\n  - url: http://a:8000\n")
    nodes = load_nodes(config_path=p)
    assert len(nodes) == 1 and nodes[0].name == "alpha" and nodes[0].token == "t1"  # dedup by url, first wins


def test_config_setting_defaults():
    from muse.core import config
    assert config.get("federation.refresh_interval_seconds") == 3.0


@pytest.mark.parametrize(
    "url",
    [
        "",
        "localhost:8000",
        "ftp://host/model",
        "http://",
        "http://user:secret@host:8000",
        "http://host:8000?mode=unsafe",
        "http://host:8000#fragment",
        " http://host:8000",
        "http://host:0",
        "http://host:invalid",
    ],
)
def test_cli_nodes_reject_ambiguous_or_unsafe_urls(url):
    with pytest.raises(ValueError, match="federation node url"):
        load_nodes(cli_nodes=[url])


@pytest.mark.parametrize(
    "entry",
    [
        {"url": 1234},
        {"url": "http://host:8000", "name": ["bad"]},
        {"url": "http://host:8000", "token": {"bad": "shape"}},
    ],
)
def test_yaml_nodes_reject_non_string_identity_fields(tmp_path, entry):
    import yaml

    path = tmp_path / "federation.yaml"
    path.write_text(yaml.safe_dump({"nodes": [entry]}))

    with pytest.raises(ValueError, match="federation node"):
        load_nodes(config_path=path)


def test_yaml_empty_token_normalizes_to_none(tmp_path):
    path = tmp_path / "federation.yaml"
    path.write_text("nodes:\n  - {url: https://host.example/base/, token: ''}\n")

    assert load_nodes(config_path=path) == [
        NodeSpec(url="https://host.example/base", name="host.example", token=None)
    ]


@pytest.mark.parametrize(
    "document",
    [
        {"nodes": {"url": "http://host:8000"}},
        {"nodes": ["http://host:8000"]},
        {"nodes": [{"name": "missing-url"}]},
    ],
)
def test_yaml_rejects_malformed_node_list_shape(tmp_path, document):
    import yaml

    path = tmp_path / "federation.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="federation .*nodes|federation node entry"):
        load_nodes(config_path=path)
