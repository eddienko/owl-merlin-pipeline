import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import dask
import numpy as np
from dask import delayed
from distributed import get_client
from owl_dev import pipeline
from owl_dev.logging import logger

from . import utils


def setup_parameters(
    dataset_name: str,
    output_dir: Path,
    analysis_parameters: Path,
    codebook: Path,
    data_organization: Path,
    microscope_parameters: Path,
    positions: Path,
):
    merged = output_dir / "merged" / dataset_name
    merged.mkdir(parents=True, exist_ok=True)

    blanks = output_dir / "blanks"
    blanks.mkdir(parents=True, exist_ok=True)

    parameters_home = output_dir / "parameters"
    parameters_home.mkdir(parents=True, exist_ok=True)

    (parameters_home / "analysis").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        analysis_parameters, parameters_home / "analysis" / analysis_parameters.name
    )

    (parameters_home / "codebooks").mkdir(parents=True, exist_ok=True)
    shutil.copy(codebook, parameters_home / "codebooks" / codebook.name)

    (parameters_home / "dataorganization").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        data_organization, parameters_home / "dataorganization" / data_organization.name
    )

    (parameters_home / "microscope").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        microscope_parameters,
        parameters_home / "microscope" / microscope_parameters.name,
    )

    (parameters_home / "positions").mkdir(parents=True, exist_ok=True)
    shutil.copy(positions, parameters_home / "positions" / positions.name)


def create_env(input_dir: Path, output_dir: Path):
    with open(output_dir / ".merlinenv", "w") as fh:
        fh.write(f"PARAMETERS_HOME={output_dir / 'parameters'}\n")
        fh.write(f"ANALYSIS_HOME={output_dir / 'analysis'}\n")
        fh.write(f"DATA_HOME={output_dir / 'merged'}\n")


def run_merlin(command: List, output_dir: Path, env=None):
    process = subprocess.Popen(
        command,
        env=env,
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip().decode())
    return process


@pipeline
def main(
    *,
    input_dir: Path,
    output_dir: Path,
    analysis_parameters: Path,
    data_organization: Path,
    codebook: Path,
    microscope_parameters: Path,
    positions: Path,
    **kwargs,
) -> int:  # pragma: no cover
    """Example pipeline.

    The pipeline constructs a list of ``datalen`` elements and
    performs a series of operations on them.

    Parameters
    ----------
    datalen
        Length of list to use.

    Returns
    -------
    result of operations
    """

    logger.info("Preprocessing")

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_parameters(
        input_dir.name,
        output_dir,
        analysis_parameters,
        codebook,
        data_organization,
        microscope_parameters,
        positions,
    )

    create_env(input_dir, output_dir)

    cmd = [
        "/home/eglez/.local/bin/merlin",
        "--generate-only",
        "-p",
        positions.name,
        "-a",
        analysis_parameters.name,
        "-m",
        microscope_parameters.name,
        "-o",
        data_organization.name,
        "-c",
        codebook.name,
        "-n",
        "10",
        input_dir.name,
    ]

    ncycles, nFOVs, nzslices, npixel = utils.inferDatasetParam(input_dir)
    logger.info(
        f"There are {ncycles} cycles, {nFOVs} FOVs, {nzslices} planes, image dimensions are {npixel}"
    )
    utils.fill_blanks(input_dir, output_dir / "blanks")

    nbitsequence = np.repeat(2, ncycles)
    fiducialplane = 1

    files = []
    for cycle in range(ncycles):
        imprefix = "merFISH__" if cycle < 10 else "merFISH_"

        files = []
        for FOV in range(nFOVs):
            imFOV = "%03d_" % FOV
            f = delayed(utils.process_FOV)(
                input_dir,
                imprefix,
                cycle,
                imFOV,
                nbitsequence,
                nzslices,
                npixel,
                fiducialplane,
                output_dir,
                FOV,
            )
            files.append(f)
        dask.compute(files)

    utils.generate_position(
        input_dir,
        os.path.join(
            output_dir / "parameters",
            "positions",
            "positions_" + input_dir.name + ".txt",
        ),
    )

    utils.generate_organization(
        input_dir,
        os.path.join(
            output_dir / "parameters",
            "dataorganization",
            "data_organization_" + input_dir.name + ".csv",
        ),
        nzslices,
    )

    shutil.rmtree(output_dir / "analysis" / input_dir.name, ignore_errors=True)

    client = get_client()
    logger.info("Running command %s ", " ".join(cmd))
    with dask.annotate(executor="processes", retries=2):
        fut = client.submit(
            run_merlin,
            cmd,
            output_dir,
            env={"MERLIN_ENV_PATH": output_dir, "PYTHONUNBUFFERED": "1"},
        )
    res = client.gather(fut)

    if res.returncode == 0:
        logger.info("Pipeline successful")
    else:
        logger.error("Pipeline failed")
        raise Exception("Pipeline failed")

    return res.returncode
