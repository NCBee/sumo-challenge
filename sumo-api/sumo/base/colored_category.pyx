#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import csv
import numpy as np
cimport numpy as np
import os


class ColoredCategory(object):
    """ Class to generate distinct RGB values for different categories of a dataset."""

    def __init__(self, csv_file):
        """ Create an object that holds color information of categories in
        <dataset_name> dataset.

        Inputs:
        csv_file (str) - filepath to an external csv file containing color mapping
            information.

        Exceptions:
        IOError - raised if csv_file does not exist.

        Note:
        The CSV file must be in the the following format:
        id,category,is_evaluated,color

        example:
        id,category,is_evaluated,color
        0,empty,False,(152, 245, 255)
        1,accordion,False,(30, 144, 255)
        2,air_conditioner,True,(122, 197, 205)
        3,amplifier,False,(110, 123, 139)
        ...
        """
        if not os.path.exists(csv_file):
            raise IOError("File {} does not exist.".format(csv_file))

        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            self._data = [{
                "category": line["category"],
                "is_evaluated": line["is_evaluated"],
                "color": tuple(map(int, line["color"].split(";")))} for line in reader]

        self._reverse_data = {
            self._data[id]["category"]: self._data[id]["color"] for id, x in enumerate(self._data)
        }
        cdef np.ndarray lut = np.zeros((0))
        self._lut = lut

    @property
    def LUT(self):
        """The lookup table for conversion of indexes to RGB values."""
        if self._lut.shape[0] == 0:
            total = len(self._data)
            self._lut = np.array([x["color"]
                for x in self._data], dtype=np.uint8)
        return self._lut

    def from_category_id_to_rgb(self, category_id):
        """ Convert a category id to its corresponding color.
        Inputs:
        category_id (int) - id of the category

        Returns
        An RGB tuple, which corresponds to the category_id

        Exceptions:
        KeyError - if category_id does not exist in the dataset config.
        """
        if category_id > len(self._data) or category_id < 0:
            raise KeyError("Category id {} does not exist.".format(category_id))
        return self._data[category_id]["color"]

    def from_category_name_to_rgb(self, category_name):
        """ Convert a category name to its corresponding color.
        Inputs:
        category_name (str) - name of the category

        Returns
        An RGB tuple, which corresponds to the category_name

        Exceptions:
        KeyError - if category_name does not exist in the dataset config.
        """
        if category_name not in self._reverse_data:
            raise KeyError("Category {} does not exist.".format(category_name))
        return self._reverse_data[category_name]

    def convert_to_rgb_im(self, np.ndarray indexed_im):
        """Coverts and indexed_im to RGB based on category mapping.
        Inputs:
        indexed_im (nxm np.array) - input indexed image.

        returns
        An RGB image (nxmx3 np.array) where each index is converted to the
        corresponding RGB value
        """
        return self.LUT[indexed_im]
