import hydra
from omegaconf import DictConfig

from src.training.models_refitting import ModelReffiter


@hydra.main(
    config_path="../configs/churn_prediction", config_name="main", version_base=None
)
def run_refit(cfg: DictConfig):
    model_definition_path_yaml = cfg.training_pipeline_settings.model_definition_yaml
    date = cfg.pipeline_settings.date

    model_instance = ModelReffiter(yaml_path=model_definition_path_yaml)

    save_path = f"""{cfg.training_pipeline_settings.output_models_directory}/
        {type(model_instance.model).__name__}_reffited.pickle"""
    features_directory = cfg.training_pipeline_settings.features_split_directory

    model_instance.refit(save_path, features_directory, date)


if __name__ == "__main__":
    run_refit()
