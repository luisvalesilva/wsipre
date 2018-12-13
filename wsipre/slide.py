# -*- coding: utf-8 -*-

"""
Handle annotated whole-slide images (WSI; also known as virtual slides).

"""

import warnings
from distutils.version import StrictVersion
import random

import numpy as np
import openslide
import cv2

import annotation as reader
import tissue


class _AnnotatedOpenSlide(openslide.OpenSlide):
    """An open *annotated* WSI.

    Wrapper of `openslide.OpenSlide` to handle annotation polygons together
    with their corresponding OpenSlide Python WSI instances.

    Parameters
    ----------
    filename: str
        Path to slide file.
    annotation_filename: str
        Path to XML annotation file.
    data_source: str
        The computational histology challenge releasing the dataset:
        * 'camelyon': CAMELYON grand challenges in pathology.
        * 'bach': BACH Grand Challenge on Breast Cancer Histology Images.

    Attributes
    ----------
    filename: str
        The WSI file name to read.
    annotation_filename: str
        Corresponding XML annotation file.
    polygons: list of lists of 2-tuples
        Polygon region annotations.
    labels: list of str
        Polygon region annotation labels.
    label_map: dict {str: int}
        Correspondence between polygon region annotation labels and integer
        values used in numpy masks.

    """

    def __init__(self, filename, annotation_filename, data_source):
        openslide.OpenSlide.__init__(self, filename)
        self.filename = filename  # Useful to name predicted annotations
        self.annotation_filename = annotation_filename
        self.data_source = data_source
        self.polygons = None
        self.label_map = None

        if self.annotation_filename is not None:
            if self.data_source == 'camelyon':
                self.polygons, self.labels = reader.camelyon_annotations(
                    self.annotation_filename)
                if 'metastases' in self.labels:
                    # Avoid value 0 (used by default for unlabeled regions)
                    self.label_map = {'metastases': 2, 'normal': 1}
                # CAMELYON16 data
                elif '_0' in self.labels or '_1' in self.labels:
                    self.label_map = {'_0': 2, '_1': 2, '_2': 1}
                else:  # Predicted annotations
                    self.label_map = {'predicted_tumor': 1}
            elif self.data_source == 'bach':
                # Value 1 is reserved for 'normal' tissue annotations
                self.label_map = {'Benign': 2, 'Carcinoma in situ': 3,
                                  'Invasive carcinoma': 4}
                self.polygons, self.labels = reader.bach_annotations(
                    self.annotation_filename)
            else:
                raise ValueError(
                    '"data_source" value must be either "camelyon" or "bach".')
        else:
            if self.data_source is not None:
                warnings.warn('"data_source" is only used if an ' +
                              '"annotation_filename" is provided.')


class Slide(_AnnotatedOpenSlide):
    """An open WSI, with or without annotations.

    Added functionality to OpenSlide WSI instances, optionally with
    annotations.

    Parameters
    ----------
    filename: str
        Path to slide file. Namesake argument to `openslide.OpenSlide` and
        `AnnotatedOpenSlide`.
    annotation_filename: str
        Path to XML annotation file. Namesake argument to `AnnotatedOpenSlide`.
    data_source: str
        The computational histology challenge releasing the dataset. Namesake
        argument to `AnnotatedOpenSlide`.

    Attributes
    ----------
    tissue_mask: Numpy 2D array
        Tissue RoI ``annotation``.
    tissue_label_map: dict {str: int}
        Correspondence between polygon region annotation labels and integer
        values used in numpy masks.
    downsampling_factor: float
        The scaling factor relative to level 0 dimensions.
    downsampled_slide: PIL image
        The scaled down slide generated when running the tissue detector.

    Examples
    --------
    >>> from wsipre import slide
    >>> wsi = slide.Slide('tumor_001.tif', 'tumor_001.xml', 'camelyon')
    >>> wsi.label_map
    {'_0': 2, '_1': 2, '_2': 1}
    >>> wsi.level_count
    10
    >>> len(wsi.polygons)
    2

    >>> thumbnail, mask, dwnspl_factor = wsi.get_thumbnail_with_annotation(
    ...     size=(3000, 3000), polygon_type='line', line_thickness=5)

    >>> slide_region = wsi.read_region_with_annotation(
    ...     location=(65000, 110000), level=4, size=(1000, 1000))

    """

    def __init__(self, filename, annotation_filename=None, data_source=None):
        self.tissue_mask = None
        self.tissue_label_map = None
        self.downsampling_factor = None
        self.downsampled_slide = None

        # OpenSlide version < 3.4.1 lacks support for some recent file formats
        version = openslide.__library_version__
        if StrictVersion(version) < StrictVersion('3.4.1'):
            warnings.warn(
                f'OpenSlide version {version} lacks support for some ' +
                'whole-slide image file formats. It is highly recommended ' +
                'to update to a more recent version.')

        if annotation_filename is None:
            if data_source is not None:
                warnings.warn(
                    '"data_source" is not used (no annotation was provided).')
            openslide.OpenSlide.__init__(self, filename)
        else:
            _AnnotatedOpenSlide.__init__(
                self, filename, annotation_filename, data_source)

    def _draw_polygons(self, mask, polygons, polygon_type,
                       line_thickness=None):
        """Convert polygon vertex coordinates to polygon drawing."""
        for poly, label in zip(polygons, self.labels):
            if polygon_type == 'line':
                mask = cv2.polylines(mask, [poly], True,
                                     int(self.label_map[label]),
                                     line_thickness)
            elif polygon_type == 'area':
                if line_thickness is not None:
                    warnings.warn('"line_thickness" is only used if ' +
                                  '"polygon_type" is "line".')

                mask = cv2.fillPoly(mask, [poly], int(self.label_map[label]))
            else:
                raise ValueError(
                    'Accepted "polygon_type" values are "line" or "area".')

        return mask

    def _draw_tissue_polygons(self, mask, polygons, polygon_type,
                              line_thickness=None):
        """Convert tissue polygon vertex coordinates to polygon drawing."""
        tissue_label = 1

        for poly in polygons:
            if polygon_type == 'line':
                mask = cv2.polylines(
                    mask, [poly], True, tissue_label, line_thickness)
            elif polygon_type == 'area':
                if line_thickness is not None:
                    warnings.warn('"line_thickness" is only used if ' +
                                  '"polygon_type" is "line".')

                mask = cv2.fillPoly(mask, [poly], tissue_label)
            else:
                raise ValueError(
                    'Accepted "polygon_type" values are "line" or "area".')

        return mask

    def _update_label_map(self, annotation_mask):
        labels = np.unique(annotation_mask)
        self.label_map = {key: label for key, label in self.label_map.items()
                          if label in labels}

        return self

    def get_thumbnail_with_annotation(self, size, polygon_type='area',
                                      line_thickness=None):
        """Convert *annotated* WSI to a scaled-down thumbnail.

        Parameters
        ----------
        size: 2-tuple
            Maximum size of compressed image thumbnail as (width, height).
        polygon_type: str
            Type of polygon drawing:
            * 'line'
            * 'area'
        line_thickness: int
            Polygon edge line thickness. Required if ``polygon_type`` is
            ``line``.

        Returns
        -------
        Scaled down slide (as PIL RGBA image) and annotation mask (as numpy 2D
        array), plus the scaling factor.

        """
        if self.polygons is None:
                raise ValueError(
                    'No annotation is available. Please load an annotation ' +
                    'or consider using "get_thumbnail" instead.')

        thumb = self.get_thumbnail(size)

        width_factor = self.dimensions[0] / size[0]
        height_factor = self.dimensions[1] / size[1]
        downsampling_factor = max(width_factor, height_factor)

        polygons = np.array([np.array(poly) for poly in self.polygons])
        polygons = polygons / downsampling_factor
        polygons = [poly.astype('int32') for poly in polygons]

        mask = np.zeros(thumb.size[::-1])  # Reverse width and height
        mask = self._draw_polygons(
            mask, polygons, polygon_type, line_thickness)

        # Drop unused labels from label_map
        # self._update_label_map(mask)

        return thumb, mask, downsampling_factor

    def read_region_with_annotation(self, location, level, size,
                                    polygon_type='area', line_thickness=None):
        """Crop a smaller region from an *annotated* WSI.

        Get a defined region from a WSI, together with its annotation mask.

        Parameters
        ----------
        location: 2-tuple
            X and y coordinates of the top left pixel. Namesake argument to
            OpenSlide's "read_region" function.
        level: int
            Slide level. Namesake argument to OpenSlide's "read_region"
            function.
        size: 2-tuple
            Crop size (width, height). Namesake argument to OpenSlide's
            "read_region" function.
        polygon_type: str
            Type of polygon drawing:
            * 'line'
            * 'area'
        line_thickness: int
            Polygon edge line thickness. Required if 'polygon_type' is 'line'.

        Returns
        -------
        Image region (as PIL RGBA image) and the annotation mask (as numpy 2d
        array).

        """
        if self.polygons is None:
                raise ValueError(
                    'No annotation is available. Please load an annotation ' +
                    'or consider using "read_region" instead.')
        downsampling_factor = self.level_downsamples[level]
        slide_region = self.read_region(location, level, size)

        polygons = np.array([np.array(poly) for poly in self.polygons])
        polygons = polygons / downsampling_factor
        polygons = [poly.astype('int32') for poly in polygons]

        location = tuple(int(coord / downsampling_factor)
                         for coord in location)

        # Convert polygon coordinates to patch coordinates
        polygons = [poly - np.array(location) for poly in polygons]

        mask_region = np.zeros(size[::-1])
        mask_region = self._draw_polygons(
           mask_region, polygons, polygon_type, line_thickness)

        # Drop unused labels from label_map
        # self._update_label_map(mask_region)

        return slide_region, mask_region

    def get_tissue_mask(self, downsampling_factor=64, polygon_type='area',
                        line_thickness=None):
        """Generate tissue region annotation.

        Make a binary mask annotating tissue regions of interest (RoI) on the
        WSI using an automatic threshold-based segmentation method inspired by
        the one used by Wang *et al.* [1]_. Briefly, the method consists on the
        following steps, starting from a WSI:
            * Select downsampling level (typically a factor of 64)
            * Transfer from the RGB to the HSV color space
            * Determine optimal threshold value in the saturation channel using
              the Otsu algorithm [2]_
            * Threshold image to generate a binary mask
            * Fill in small holes and remove small objects

        .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad,
           Andrew H. Beck, "Deep Learning for Identifying Metastatic Breast
           Cancer", arXiv:1606.05718.

        .. [2] Nobuyuki Otsu, "A Threshold Selection Method from Gray-level
           Histograms", IEEE Trans Syst Man Cybern., 9(1):62–66, 1979.

        Parameters
        ----------
        downsampling_factor: int
            The desired factor to downsample the image by, since full WSIs will
            not fit in memory. The image's closest level downsample is found
            and used.
        polygon_type: str
            Type of polygon drawing:
            * 'line'
            * 'area'
        line_thickness: int
            Polygon edge line thickness. Required if ``polygon_type`` is
            ``line``.

        Returns
        -------
        Binary mask as numpy 2D array, RGB slide image (in the used
        downsampling level, to allow visualization) and downsampling factor.

        """
        (mask_contours,
         self.downsampled_slide,
         self.downsampling_factor) = tissue.detect_tissue(
             self, downsampling_factor)

        mask = np.zeros(self.downsampled_slide.shape[:2])
        self.tissue_mask = self._draw_tissue_polygons(
           mask, mask_contours, polygon_type, line_thickness)
        self.tissue_label_map = {'background': 0, 'tissue': 1}

        return self

    def _get_random_coordinates(self, level, annotation_mask,
                                mask_downsampling_factor, target_class, size):
        """Get region (patch) coordinates at random."""
        # Select coordinates matching target class at random
        pixels_in_roi = tuple(zip(*np.where(annotation_mask == target_class)))

        # TODO: implement way to avoid tumor (or other label) coordinates:
        # add argument labels to avoid (e.g. avoid_labels=[1, 2])
        # check that annotations are available
        # generate annotation thumbnail of the same size as tissue mask
        # remove intersection of pixels_in_roi with pixels of coordinates found
        #     in "avoid_labels"

        coordinates = random.choice(pixels_in_roi)

        # Scale coordinates up to level-0 dimensions
        coordinates = [int(x * mask_downsampling_factor) for x in coordinates]

        # OpenSlide Python's read_region takes top left corner position, which
        # effectively excludes tissue above and to the left of annotations
        # Fix it by offsetting coordinates: convert to center pixel position
        # (subtract half of patch size from x and y coordinates)
        half_size = [x * self.level_downsamples[level] / 2 for x in size]
        row_location = coordinates[0] - half_size[0]
        col_location = coordinates[1] - half_size[1]

        # Make sure coordinates are still within slide margins
        cols_edge = self.dimensions[0] - size[0]
        rows_edge = self.dimensions[1] - size[1]
        row_location = max(0, min(row_location, rows_edge))
        col_location = max(0, min(col_location, cols_edge))

        return int(col_location), int(row_location)  # OpenSlide: width, height

    def _max_repeated_pixel_ratio(self, image):
        """Compute ratio of count of most common pixel value to total count."""
        image = np.array(image)
        _, counts = np.unique(image, return_counts=True)

        return np.max(counts) / image.size

    def read_random_tissue_patch(self, level, size):
        """Crop random patch from detected tissue on WSI.

        Parameters
        ----------
        level: int
            Slide level. Namesake argument to OpenSlide's "read_region"
            function.
        size: 2-tuple
            Crop size (width, height). Namesake argument to OpenSlide's
            "read_region" function.

        Returns
        -------
        Image region or patch (as PIL RGBA image).

        """
        if self.tissue_mask is None:
            self.get_tissue_mask()

        coordinates = self._get_random_coordinates(
            level, self.tissue_mask, self.downsampling_factor, 1, size)
        patch = self.read_region(coordinates, level, size)

        # Avoid false positive regions where a large proportion of pixels is
        # exactly the same
        while self._max_repeated_pixel_ratio(patch) > 0.5:
            coordinates = self._get_random_coordinates(
                level, self.tissue_mask, self.downsampling_factor, 1, size)
            patch = self.read_region(coordinates, level, size)

        return patch

    def _get_area_ratio(self, mask, target_class):
        """Compute ratio of pixels labeled as 'target_class' to all pixels."""
        class_pixels = mask[np.where(mask == target_class)].size

        return class_pixels / mask.size

    def _pick_random_coordinates(self, level, size, target_class):
        """Pick random patch to crop from WSI."""
        # Get thumbnail and select coordinates at random
        _, mask, downsampling_factor = self.get_thumbnail_with_annotation(
             size=(5000, 5000), polygon_type='area')
        coordinates = self._get_random_coordinates(
            level, mask, downsampling_factor, target_class, size)

        # Get ratio (polygon type must be 'area')
        _, mask = self.read_region_with_annotation(
            coordinates, level, size, polygon_type='area')
        area_ratio = self._get_area_ratio(mask, target_class)

        return coordinates, area_ratio

    def read_random_patch(self, level, size, target_class,
                          min_class_area_ratio, polygon_type='area',
                          line_thickness=None):
        """Crop random patch from WSI.

        Select random location within target class according to provided
        annotation.

        Parameters
        ----------
        level: int
            Slide level. Namesake argument to OpenSlide's "read_region"
            function.
        size: 2-tuple
            Crop size (width, height). Namesake argument to OpenSlide's
            "read_region" function.
        target_class: int
            The class annotation of the central pixel of the patch.
            function.
        min_class_area_ratio: float (0, 1]
            Minimum ratio of target class pixels to total pixels.
        polygon_type: str
            Type of polygon drawing:
            * 'line'
            * 'area'
        line_thickness: int
            Polygon edge line thickness. Required if ``polygon_type`` is
            ``line``.

        Returns
        -------
        Image region or patch (as PIL RGBA image).

        """
        if not 0 < min_class_area_ratio <= 1:
            raise ValueError(
                '"min_class_area_ratio" must be in the interval (0, 1].')

        coordinates, area_ratio = self._pick_random_coordinates(
            level, size, target_class)

        # When the slide level is high, the following while loop may be
        # infinite, since the tumor area will never be high enough.
        # Try to sample a few times and throw error if not successful
        i = 0
        while area_ratio < min_class_area_ratio:
            i += 1
            coordinates, area_ratio = self._pick_random_coordinates(
                level, size, target_class)
            if i > 15:
                raise ValueError(
                    'Cannot seem to find patch with a minimum of ' +
                    f'{min_class_area_ratio * 100}% of class {target_class}.' +
                    ' The chosen slide level may be too high.')

        # Get actual data (chosen polygon type)
        patch, mask = self.read_region_with_annotation(
            coordinates, level, size, polygon_type, line_thickness)

        return patch, mask
