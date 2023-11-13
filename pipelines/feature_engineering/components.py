import os
from typing import Optional

import click
from omegaconf import DictConfig, OmegaConf

from src.feature_engineering.etl import load_data
from src.feature_engineering.model_specific_transformations import (
    apply_one_hot_encoding,
    encode_categorical_features,
    log_transform_features,
    replace_values,
)
from src.utils import save_dataframe
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
    "-p", "--input_path", type=str, default=None, help="Path to the data file"
)
@click.option(
    "-o",
    "--output_path",
    type=str,
    default=None,
    help="Directory to save datasets",
)
@click.option("-dt", "--date", type=str, default=None, help="Date to save datasets")
def run_feature_engineering(
    ctx: click.Context,
    input_path: Optional[str],
    output_path: Optional[str],
    date: Optional[str],
) -> None:
    """
    Run feature engineering process, save resulting datasets to specified directory.
    """
    cfg = ctx.obj.cfg

    # Validate and/or set default values from the configuration
    input_path = input_path or cfg.fe_pipeline_settings.input_path
    if not input_path:
        logger.error(
            "Input path is neither provided as an argument nor found in the configuration."
        )
        return

    output_path = output_path or cfg.fe_pipeline_settings.output_path
    if not output_path:
        logger.error(
            "Output directory is neither provided as an argument nor found in the configuration."
        )
        return

    date = date or cfg.pipeline_settings.date
    if not date:
        logger.warning(
            "Date is neither provided as an argument nor found in the configuration. Defaulting to 'None'."
        )

    logger.info(f"Running feature engineering with data from {input_path}")

    # Load and preprocess data
    df = load_data(input_path)
    transform_log_features = [
        "Credit_Limit",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    df = log_transform_features(df, transform_log_features)

    transform_categorical_features = ["Gender", "Attrition_Flag"]
    df = encode_categorical_features(df, transform_categorical_features)

    transform_one_hot_encoding_features = ["Card_Category", "Marital_Status"]
    df = apply_one_hot_encoding(df, transform_one_hot_encoding_features)

    replace_struct = {
        "Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1},
        "Education_Level": {
            "Doctorate": 5,
            "Post-Graduate": 4,
            "Graduate": 3,
            "College": 2,
            "High School": 1,
            "Unknown": 0,
            "Uneducated": -1,
        },
        "Income_Category": {
            "$120K +": 4,
            "$80K - $120K": 3,
            "$60K - $80K": 2,
            "$40K - $60K": 1,
            "Unknown": 0,
            "Less than $40K": -1,
        },
    }
    df = replace_values(df, replace_struct)

    save_dataframe(df, path=output_path)


@click.option("--config", type=str, default=None, help="Path to the configuration file")
def main(config: Optional[str] = None) -> None:
    """Entry point of the script."""
    if not config:
        config = "pipelines/configs/churn_prediction/main.yaml"
    if not os.path.exists(config):
        raise ValueError(f"Configuration file not found at {config}")

    cfg = OmegaConf.load(config)
    cli(obj=Context(cfg))


if __name__ == "__main__":
    main()
