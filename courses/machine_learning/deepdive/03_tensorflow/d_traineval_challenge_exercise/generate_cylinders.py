import argparse
from math import pi
import numpy as np
import os
import pandas as pd

MIN_RADIUS = 0.5
MAX_RADIUS = 2.0
MIN_HEIGHT = 0.5
MAX_HEIGHT = 2.0
ADDED_ERROR = 0.1
NUM_DECIMALS = 1


def generate_cylinder_df(size):
    """Generate a dataframe of cylinders where the radius and height of
    each cylinder are both in the range 0.5 to 2.0 and the volume equals
    (pi * r^2) * h. Then add up to a 10% error (uniformly distributed) to
    the volume, followed by rounding off radius, height and volume to the
    nearest 0.1.
    """

    # Generate radiuses and heights
    radius = np.random.uniform(MIN_RADIUS, MAX_RADIUS, size=size)
    height = np.random.uniform(MIN_HEIGHT, MAX_HEIGHT, size=size)

    # Calculate the correct volumes with those radiuses and heights
    volume = (pi * radius ** 2) * height

    # Add the error to the volumes
    volume = volume * np.random.uniform(
        1 - ADDED_ERROR,
        1 + ADDED_ERROR,
        size=size)

    # Then round off radius, height and volume
    radius = np.round(radius, decimals=NUM_DECIMALS)
    height = np.round(height, decimals=NUM_DECIMALS)
    volume = np.round(volume, decimals=NUM_DECIMALS)

    df = pd.DataFrame({
        'volume': volume,
        'radius': radius,
        'height': height, })

    return df


if __name__ == '__main__':
    if ('get_ipython' not in dir()) & ('PYCHARM_HOSTED' not in os.environ):
        # i.e. if run from the command line

        # Handle command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--filename',
            type=str,
            help='the filename to save the cylinders to',
            required=True)
        parser.add_argument(
            '--size',
            type=int,
            help='the number of cylinders to generate',
            required=True)
        # parser.add_argument(
        #     '--job-dir',
        #     help='this model ignores this field, but it is required by gcloud',
        #     default='junk')
        args = parser.parse_args()
        arguments = args.__dict__
        # arguments.pop('job_dir', None)

        # Generate cylinders and write them to file
        generate_cylinder_df(arguments['size']).to_csv(
            arguments['filename'],
            index=False)

        print('saved {} cylinders to {}'.format(
            arguments['size'],
            arguments['filename']))

    else:  # if run from a notebook or IDE
        files_to_generate = {
            'input/cylinders_train.csv': 8000,
            'input/cylinders_eval.csv': 1000,
            'input/cylinders_test.csv': 1000,}

        for filename, size in files_to_generate.items():
            generate_cylinder_df(size=size).to_csv(
                filename,
                index=False)

            print('saved {} cylinders to {}'.format(
                size,
                filename))
