"""Writes reference decision variable values into a file as a list.

Will write into a file 'dec_vars.json' in the given directory.

Arguments:
    -d: Data directyory. The directory that has the different forests' data in their corresponding directories.
        Assumes that in the directory there exists a file 'reference_solutions.json' that has all
        the different forests' solutions that are to be used as a reference in the UTOPIA problem.
        Defaults to 'C:/MyTemp/code/UTOPIA/alternatives'.
    -f: Forest name. The name of the forest whose decision variables are to be written in a file.
        Assumes that a directory of this name exists in the directory given in the '--dir' argument.
        Defaults to 'select'.
"""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="dir", default="C:/MyTemp/code/UTOPIA/alternatives")
    parser.add_argument("-f", dest="forest_name", default="select")
    args = parser.parse_args()
    data_dir = args.dir
    forest_name = args.forest_name

    with Path.open(f"{data_dir}/reference_solutions.json") as file:
        d = json.load(file)
        dec_vars = json.loads(d[forest_name]["decisions_vars"])
    baseline = []
    for var in dec_vars:
        if "X" in var:
            baseline.append(dec_vars[var])
    with Path.open(f"{data_dir}/{forest_name}/dec_vars.json", "w") as f:
        json.dump(baseline, f)
