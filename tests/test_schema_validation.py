from am_mvt.config import load_config


def test_config_contains_project_title():
    config = load_config()
    assert "project" in config
    assert "title" in config["project"]
    assert "Additive Manufacturing" in config["project"]["title"]


def test_config_contains_schema_variables():
    config = load_config()
    assert "schema" in config
    assert "input_variables" in config["schema"]
    assert "output_variables" in config["schema"]
    assert "alloy" in config["schema"]["input_variables"]
    assert "uts_MPa" in config["schema"]["output_variables"]
    assert "failure_mode" not in config["schema"]["output_variables"]
