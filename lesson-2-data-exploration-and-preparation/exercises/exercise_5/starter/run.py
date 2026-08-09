#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # create the run
    run = wandb.init(project="exercise_5", job_type="process_data")

    # declaring that I'm using the artifact provided as input
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    #.file() downloads the file and artifact_path is my local path variable i created
    artifact_path = artifact.file()

    # read in the artifact
    df = pd.read_parquet(artifact_path)

    # Pre-processing
    # Drop the duplicates and reset the index so there aren't a bunch of empties
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info("Fixing missing values")
    # These are missing values that are due to an old version of the data. On new data,
    # because of a change in the web form used to register new songs, the title and the
    # song name are already empty strings
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    # create new column by concatenating the title and song name
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # save file to a new variable
    outfile = args.artifact_name
    # saving locally to a csv file
    df.to_csv(outfile)

    # creating empty shell of an artifact
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    # attaching artifact file so its uploaded as part of the run
    artifact.add_file(outfile)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    #no need to call run.finish() if there's only one run
    os.remove(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)