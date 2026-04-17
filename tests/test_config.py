from gencast_repro.config import load_experiment_config


def test_load_mini_config():
    config = load_experiment_config("configs/mini.yaml")
    assert config.name == "gencast-mini"
    assert config.model.input_steps == 2

