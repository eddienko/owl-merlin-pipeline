import tifffile as tf
import numpy as np
import pandas as pd
import os
import re

from owl_dev.logging import logger

MATCH = re.compile(r"(\d+_\d+_\d+)")


def inferDatasetParam(imagedir):
    """Calculate the dimensions of the merFISH experiment"""

    def _info(val):
        return MATCH.search(val.name).groups()[0]

    image_list = [*imagedir.glob("*.tif")]
    image_list = pd.Series(image_list)
    image_dim = tf.imread(f"{image_list[0]}").shape[1:]
    image_df = image_list.apply(_info).str.split("_", expand=True)
    return (
        len(image_df[0].unique()),
        len(image_df[1].unique()),
        len(image_df[2].unique()),
        image_dim,
    )


# Find missing images and fill in with 0-value images for 1st (647-nm) and 2nd (568-nm) channels, and combine with the beads (3rd) channel image from the corresponding tiff-stack from a nearest cycle
def fill_blanks(inpath, outpath=None):
    ncycles, nFOVs, nzslices, npixel = inferDatasetParam(inpath)

    outpath = outpath or os.path.join(inpath, "blanks")
    os.makedirs(outpath, exist_ok=True)
    blank = np.ndarray(shape=(1, *npixel), dtype="uint16")
    blank_counter = 0

    with open(os.path.join(outpath, "blank_log.txt"), "w") as outfile:

        for cycle in range(ncycles):
            if cycle < 10:
                imprefix = "merFISH__"
            else:
                imprefix = "merFISH_"

            for FOV in range(nFOVs):
                imFOV = "%03d_" % FOV
                for z in range(1, nzslices + 1):
                    imZ = "%02d.tif" % z
                    file_to_find = os.path.join(
                        inpath, imprefix + str(cycle).zfill(2) + "_" + imFOV + imZ
                    )
                    if os.path.isfile(file_to_find):
                        pass
                    else:
                        outfile.write(f"missing file {file_to_find}\n")
                        blank_counter += 1
                        if os.path.isfile(
                            os.path.join(
                                inpath,
                                imprefix + str(cycle - 1).zfill(2) + "_" + imFOV + imZ,
                            )
                        ):
                            found_image = str(
                                os.path.join(
                                    inpath,
                                    imprefix
                                    + str(cycle - 1).zfill(2)
                                    + "_"
                                    + imFOV
                                    + imZ,
                                )
                            )
                            outfile.write(f"   File found {found_image}\n")
                            target = inpath
                            surrocycle = cycle - 1
                        elif os.path.isfile(
                            os.path.join(
                                inpath,
                                imprefix + str(cycle + 1).zfill(2) + "_" + imFOV + imZ,
                            )
                        ):
                            found_image = str(
                                os.path.join(
                                    inpath,
                                    imprefix
                                    + str(cycle + 1).zfill(2)
                                    + "_"
                                    + imFOV
                                    + imZ,
                                )
                            )
                            outfile.write(f"   File found {found_image}\n")
                            target = inpath
                            surrocycle = cycle + 1
                        else:
                            target = outpath

                        surrogate = np.concatenate(
                            (
                                blank,
                                blank,
                                tf.imread(
                                    os.path.join(
                                        target,
                                        imprefix
                                        + str(surrocycle).zfill(2)
                                        + "_"
                                        + imFOV
                                        + imZ,
                                    )
                                )[[-1]],
                            )
                        )
                        tf.imsave(
                            os.path.join(
                                outpath,
                                imprefix + str(surrocycle).zfill(2) + "_" + imFOV + imZ,
                            ),
                            data=surrogate,
                        )

    logger.info(f"The script identified {blank_counter} blank images")


# Processes each FOV by stacking each Z slice for the two channels containing the bit images, and adding the bead images
# (only one reference Z) at the end
# If there is flat-fielding this would likely happen here, after a reference image has been produced for each channel
def process_FOV(
    dataset,
    imprefix,
    cycle,
    imFOV,
    nbitsequence,
    nzslices,
    npixel,
    fiducialplane,
    outpath,
    FOV,
):
    merged = np.ndarray(
        shape=(nbitsequence[cycle] * nzslices + 1, *npixel), dtype="uint16"
    )

    for z in range(1, nzslices + 1):
        imagename = dataset / (imprefix + str(cycle).zfill(2) + "_" + imFOV + "%02d.tif" % z)
        if imagename.exists():
            image = tf.imread(f"{imagename}")[0:2]
            # flat-fielding would happen here
            merged[(z - 1) * 2 : z * 2] = image
            if z == fiducialplane:
                merged[-1] = tf.imread(f"{imagename}")[2]
        else:
            try:
                image_to_use = outpath / "blanks" / imagename.name
                image = tf.imread(image_to_use)[0:2] # get from blank folder if image is missing
                # no flat fielding here as the image would be blank and it would likely introduce errors
                merged[(z - 1) * 2 : z * 2] = image
                # outfile.write(f"Using image {image_to_use}  ")
                if z == fiducialplane:
                    merged[-1] = tf.imread(f"{image_to_use}")[2]
            except:
                logger.warning("Image %s not found" % imagename.name)

    tf.imsave(
        os.path.join(outpath, "merged", dataset.name, "merFISH_merged_%02d_%03d.tif" % (cycle, FOV)),
        data=merged,
    )


# Mock function so dask delayed has an ending point to converge to (might not be needed!)
def collect_results(files):
    return files


# Converts the stage coordinate files in a merlin position file
def generate_position(inpath, outfilepath):

    with open(os.path.join(inpath, "stage_log.txt"), "r") as infile:
        with open(outfilepath, "w") as outfile:
            r = infile.read()
            scan_chunk = r.split("stack")[0]
            other_chunk = r.split("stack")[1]
            lines = scan_chunk.split("\n")

            fov_coords = []
            for n in range(2, len(lines), 3):
                fields = dict()
                line = lines[n]
                # print(line)
                fields["x"] = line.split(",")[0].strip()
                fields["y"] = line.split(",")[1].strip()
                # print(fields)
                fov_coords.append(fields)

            for n3 in range(0, len(fov_coords)):
                stage_string = fov_coords[n3]["x"] + "," + fov_coords[n3]["y"] + "\n"
                # print(stage_string)
                outfile.write(stage_string)


# Extracts the z coordinates of each plane
def extract_z_position(inpath):
    positions = []
    with open(inpath, "r") as infile:
        r = infile.read()
        z_scan_chunk = r.split("stack")[1].split("return")[0]
        lines = z_scan_chunk.split("\n")
        for line in lines[2:-1]:
            zpos = line.split(",")[2].strip(" ")
            positions.append(float(zpos))
    return positions


# Generates the data organization file from the data_organization.csv provided into the raw data folder
def generate_organization(inpath, outfilepath, nzslices):
    data_structure = pd.read_csv(os.path.join(inpath, "data_organization.csv"))
    new_df = pd.DataFrame()
    bitn = 1
    for index, row in data_structure.iterrows():
        data = list()
        data.append(row.readoutName)
        data.append(row.channelName)
        data.append("merFISH_merged")
        data.append("(?P<imageType>[\w|-]+)_(?P<imagingRound>[0-9]+)_(?P<fov>[0-9]+)")
        data.append(bitn)
        data.append(row.imagingRound)
        data.append(row.color)
        if row.color == 650:
            frame = list(range(0, nzslices * 2, 2))
        elif row.color == 568:
            frame = list(range(1, nzslices * 2, 2))
        data.append(frame)
        zplanes = extract_z_position(os.path.join(inpath, "stage_log.txt"))
        data.append(zplanes)
        data.append("merFISH_merged")
        data.append("(?P<imageType>[\w|-]+)_(?P<imagingRound>[0-9]+)_(?P<fov>[0-9]+)")
        data.append(row.imagingRound)
        data.append(nzslices * 2)
        data.append(row.fiducialColor)
        row_series = pd.Series(data)
        row_df = pd.DataFrame([row_series], index=[bitn])
        new_df = pd.concat([new_df, row_df])
        bitn = bitn + 1
    new_df.columns = [
        "readoutName",
        "channelName",
        "imageType",
        "imageRegExp",
        "bitNumber",
        "imagingRound",
        "color",
        "frame",
        "zPos",
        "fiducialImageType",
        "fiducialRegExp",
        "fiducialImagingRound",
        "fiducialFrame",
        "fiducialColor",
    ]
    new_df.to_csv(outfilepath, index=False)
