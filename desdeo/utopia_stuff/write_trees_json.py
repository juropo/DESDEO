"""Writes the stands' tree information into a json file.

Arguments:
    -d: Data directory. The directory that has the holding's data. Assumes that the directory has the file 'trees.txt'.
        Defaults to 'C:/MyTemp/code/UTOPIA/alternatives/select'.
"""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="dir", default="C:/MyTemp/code/UTOPIA/alternatives/select")
    args = parser.parse_args()
    data_dir = args.dir

    # read tree data from file to python lists to be looped through
    line_lengths = []
    lines = []
    with Path.open(f"{data_dir}/trees.txt") as file:
        i = 0
        for line in file:
            line_lengths.append([i, len(line.split())])
            lines.append(line.split())
            i = i + 1

    # determine where stand changes (marked by two consecutive empty lines)
    double_empty_lines = [
        line_lengths[i][0] for i in range(1, len(line_lengths)) if line_lengths[i][1] == 0 and line_lengths[i-1][1] == 0
    ]

    # get a list of the stands' contents by using the indices of the double empty lines
    stands = []
    stands.append(lines[:double_empty_lines[0]])
    for i in range(1, len(double_empty_lines)):
        stands.append(lines[double_empty_lines[i-1]+1:double_empty_lines[i]])

    # loop through the stands to form a dict of the stands with standid as the stands' identifiers
    # will be a dict of the tree information of each year of each schedule of each stand
    # probably could be done a lot more efficiently
    trees = {}
    for stand in stands:
        # determine where schedule changes within the stand
        empty_lines = [i for i in range(len(stand)) if len(stand[i]) == 0]
        schedules = []
        # get a list of schedules' contents by using the indices of the empty lines
        if len(empty_lines) > 0:
            schedules.append(stand[:empty_lines[0]])
            for i in range(1, len(empty_lines)):
                schedules.append(stand[empty_lines[i-1]+1:empty_lines[i]])
        # loop through the schedules to form a dict of the schedules
        schedules_dict = {}
        for s in schedules:
            len_year_line = 3
            # determine where the year changes
            new_years = [i for i in range(len(s)) if len(s[i]) == len_year_line]
            years = []
            # get a list of the different years' by using the indices that determine new year lines
            for i in range(1, len(new_years)):
                years.append(s[new_years[i-1]:new_years[i]])
            years.append(s[new_years[len(new_years)-1]:])
            # loop through the years to form a dict of the tree information of each year
            year_dict = {}
            for y in years:
                tree_dict = {}
                for line in y:
                    if len(line) == len_year_line:
                        year = int(line[2])
                    else:
                        tree_dict[int(line[1])] = float(line[5])
                    year_dict[year] = tree_dict
            schedules_dict[int(s[0][1])] = year_dict
        trees[float(stand[0][0])] = schedules_dict

    with Path.open(f"{data_dir}/trees.json", "w") as f:
        json.dump(trees, f)
