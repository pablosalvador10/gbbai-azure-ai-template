from typing import Literal, Optional

import click
from omegaconf import DictConfig, OmegaConf

from pipelines.configs.churn_prediction import utils
from src.training.models_training import ModelTrainer
from src.training.utils import load_datasets, save_datasets
from src.utils import save_model_to_pickle
from utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


class Context:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg


@click.group()
@click.pass_context
def cli(ctx) -> None:
    """Execute before every command."""
    logger.info("Executing the pipeline component...")


@cli.command()
@click.pass_context
@click.option(
    "--features-path", type=str, help="Path to the features dataset.", default=None
)
@click.option(
    "--target-column", type=str, help="Name of the target column.", default=None
)
@click.option(
    "--estimator",
    type=str,
    required=True,
    help="Classifier to use.",
)
@click.option(
    "--train-scoring-split", type=float, default=0.2, help="Train-test split ratio."
)
@click.option(
    "--random-state", type=int, default=555, help="Random state for reproducibility."
)
@click.option(
    "--perform-sampling-techniques",
    type=click.Choice(["upsampling", "undersampling"]),
    default=None,
    help="Sampling technique to apply.",
)
@click.option(
    "--output-directory",
    type=str,
    default=None,
    help="Directory to save the resultant datasets.",
)
@click.option(
    "--date", type=str, default=None, help="Date associated with the operation."
)
def run_training_data_prep(
    ctx: click.Context,
    estimator: str,
    features_path: Optional[str],
    target_column: Optional[str],
    train_scoring_split: Optional[float] = 0.2,
    random_state: int = 555,
    perform_sampling_techniques: Optional[
        Literal["upsampling", "undersampling"]
    ] = None,
    output_directory: Optional[str] = None,
    date: Optional[str] = None,
) -> None:
    """
    Prepare training data by performing feature engineering, sampling techniques if specified,
    and saving the resultant datasets to the designated output directory.
    """

    cfg = ctx.obj.cfg

    # Validate and set defaults using configuration if arguments are not provided
    features_path = features_path or cfg.training_pipeline_settings.features_path
    if not features_path:
        print("features_path")
        logger.error("Features path is not provided or found in the configuration.")
        return

    output_directory = (
        output_directory or cfg.training_pipeline_settings.output_directory
    )
    if not output_directory:
        print("output_directory")
        logger.error("Output directory is not provided or found in the configuration.")
        return

    target_column = target_column or cfg.training_pipeline_settings.target_column
    if not target_column:
        print("target_column")
        logger.error("Target column is not provided or found in the configuration.")
        return

    train_scoring_split = (
        train_scoring_split
        or cfg.training_pipeline_settings.hyper_parameter_optimization.models.estimator.estimator_split.train_scoring_split
    )
    if train_scoring_split is None:
        print("train_scoring_split")
        logger.error(
            "Train scoring split ratio is not provided or found in the configuration."
        )
        return

    date = date or cfg.pipeline_settings.date
    if not date:
        print("date")
        logger.warning(
            "Date is not provided or found in the configuration. Defaulting to 'None'."
        )

    logger.info(f"Running feature engineering with data from {features_path}")
    formatted_estimator = estimator.replace(
        " ", "_"
    )  # Example formatting, adjust as needed
    model_type = cfg.training_pipeline_settings.hyper_parameter_optimization.models.get(
        formatted_estimator, {}
    ).get("model_type", None)

    modeltrainer_instance = ModelTrainer(  # type: ignore
        features_path, estimator=model_type, random_state=random_state
    )

    if perform_sampling_techniques:
        if perform_sampling_techniques == "upsampling":
            strategy = (
                cfg.training_pipeline_settings.hyper_parameter_optimization.models.get(
                    formatted_estimator, {}
                )
                .get("estimator_upsampling", {})
                .get("strategy", None)
            )

            k_neighbors = (
                cfg.training_pipeline_settings.hyper_parameter_optimization.models.get(
                    formatted_estimator, {}
                )
                .get("estimator_upsampling", {})
                .get("k_neighbors", None)
            )

            if not strategy or not k_neighbors:
                logger.error(
                    "Required parameters for upsampling are missing from the configuration."
                )
                return

            X_train_res, y_train_res = modeltrainer_instance.perform_upsampling(
                target_column=target_column, strategy=strategy, k_neighbors=k_neighbors
            )
            X_train, X_test, y_train, y_test = modeltrainer_instance.split_data(
                X=X_train_res, y=y_train_res, test_size=train_scoring_split
            )

        elif perform_sampling_techniques == "undersampling":
            # Logic for undersampling goes here
            pass
    else:
        X_train, X_test, y_train, y_test = modeltrainer_instance.split_data(
            target_column=target_column, test_size=train_scoring_split
        )

    save_datasets(X_train, X_test, y_train, y_test, output_directory, date)


@cli.command()
@click.pass_context
@click.option(
    "--features_directory", type=str, help="Path to the features dataset.", default=None
)
@click.option(
    "--estimator",
    type=str,
    required=True,
    help="Classifier to use.",
)
@click.option(
    "--random-state", type=int, default=555, help="Random state for reproducibility."
)
@click.option(
    "--output_models_directory",
    type=str,
    default=None,
    help="Directory to save the resultant datasets.",
)
@click.option(
    "--date", type=str, default=None, help="Date associated with the operation."
)
def run_hyper_parameter_opt(
    ctx: click.Context,
    estimator: Optional[str] = None,
    features_directory: Optional[str] = None,
    output_models_directory: Optional[str] = None,
    random_state: int = 555,
    date: Optional[str] = None,
) -> None:
    """
    Prepare training data by performing feature engineering, sampling techniques if specified,
    and saving the resultant datasets to the designated output directory.
    """

    cfg = ctx.obj.cfg

    formatted_estimator = estimator.replace(" ", "_")  # type: ignore

    # Fetch the model type
    model_type = cfg.training_pipeline_settings.hyper_parameter_optimization.models.get(
        formatted_estimator, {}
    ).get("model_type", None)

    # If the model_type doesn't exist, raise an error or handle accordingly
    if not model_type:
        logger.error(f"Model type not found for estimator: {formatted_estimator}")
        return

    # Fetch the hyperparameter optimization configurations
    estimator_parameters_opt = (
        cfg.training_pipeline_settings.hyper_parameter_optimization.models.get(
            formatted_estimator, {}
        ).get("estimator_parameters_opt", {})
    )

    scorer = estimator_parameters_opt.get("scorer", None)
    n_jobs = estimator_parameters_opt.get("n_jobs", None)
    n_iter_search = estimator_parameters_opt.get("n_iter_search", None)
    apply_pca = estimator_parameters_opt.get("apply_pca", None)

    # Validate and set defaults using configuration if arguments are not provided
    features_directory = (
        features_directory or cfg.training_pipeline_settings.features_split_directory
    )
    if not features_directory:
        logger.error("Features path is not provided or found in the configuration.")
        return

    output_models_directory = (
        output_models_directory
        or cfg.training_pipeline_settings.output_models_directory
    )
    if not output_models_directory:
        logger.error("Output directory is not provided or found in the configuration.")
        return

    date = date or cfg.pipeline_settings.date
    if not date:
        logger.warning(
            "Date is not provided or found in the configuration. Defaulting to 'None'."
        )

    logger.info(f"Running feature engineering with data from {features_directory}")

    X_train, X_test, y_train, y_test = load_datasets(
        directory=features_directory, date=date
    )

    modeltrainer_instance = ModelTrainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        estimator=model_type,
        random_state=random_state,
    )

    tuned_estimator = modeltrainer_instance.run_hyperparameter_opt(
        scorer=scorer,
        n_jobs=n_jobs,
        n_iter_search=n_iter_search,
        apply_pca=apply_pca,
        parameters=utils.parameters_train_ada_boost_upsampling,
    )

    output_models_directory = output_models_directory + f"{estimator}" + ".pickle"

    save_model_to_pickle(tuned_estimator, output_models_directory)


@click.option(
    "--config_main", type=str, default=None, help="Path to the configuration file"
)
@click.option(
    "--config_hyper",
    type=str,
    default=None,
    help="Path to the hyper-parameter configuration file",
)
def main(config_main: Optional[str] = None, config_hyper: Optional[str] = None) -> None:
    """Entry point of the script."""
    if not config_main:
        config_main = "pipelines/configs/churn_prediction/main.yaml"
    if not config_hyper:
        config_hyper = "pipelines/configs/churn_prediction/hyper_parameter_opt.yaml"

    # Load the hyper parameter configuration
    cfg = OmegaConf.load(config_main)
    hyper_param_cfg = OmegaConf.load(config_hyper)

    # Set the content of the hyper parameter file in the main configuration
    cfg.training_pipeline_settings.hyper_parameter_optimization = hyper_param_cfg

    cli(obj=Context(cfg))


if __name__ == "__main__":
    main()
