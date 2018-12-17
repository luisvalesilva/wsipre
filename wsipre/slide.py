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
    xml_style: str
        The annotation XML style. Typically associated with a computational
        histology challenge releasing the dataset.
        * 'asap': CAMELYON grand challenges in pathology.
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

    def __init__(self, filename, annotation_filename, xml_style):
        openslide.OpenSlide.__init__(self, filename)
        self.filename = filename  # Useful to name predicted annotations
        self.annotation_filename = annotation_filename
        self.xml_style = xml_style
        self.polygons = None
        self.label_map = None

        if self.annotation_filename is not None:
            if self.xml_style == 'asap':
                self.polygons, self.labels = reader.asap_annotations(
                    self.annotation_filename)
                # CAMELYON17 data
                if 'metastases' in self.labels:
                    # Avoid value 0 (used by default for unlabeled regions)
                    self.label_map = {'metastases': 2, 'normal': 1}
                # CAMELYON16 data
                elif '_0' in self.labels or '_1' in self.labels:
                    self.label_map = {'_0': 2, '_1': 2, '_2': 1}
                else:  # Predicted annotations
                    self.label_map = {'predicted_tumor': 1}
            elif self.xml_style == 'bach':
                # Value 1 is reserved for 'normal' tissue annotations
                self.label_map = {'Benign': 2, 'Carcinoma in situ': 3,
                                  'Invasive carcinoma': 4}
                self.polygons, self.labels = reader.bach_annotations(
                    self.annotation_filename)
            else:
                raise ValueError(
                    '"xml_style" value must be either "asap" or "bach".')
        else:
            if self.xml_style is not None:
                warnings.warn('"xml_style" is only used if an ' +
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
    xml_style: str
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
    >>> wsi = slide.Slide('tumor_001.tif', 'tumor_001.xml', 'asap')
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

    def __init__(self, filename, annotation_filename=None, xml_style=None):
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
            if xml_style is not None:
                warnings.warn(
                    '"xml_style" is not used (no annotation was provided).')
            openslide.OpenSlide.__init__(self, filename)
        else:
            _AnnotatedOpenSlide.__init__(
                self, filename, annotation_filename, xml_style)

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

    def _get_random_coordinates(self, pixels_in_roi, level,
                                mask_downsampling_factor, size):
        """Get region (patch) coordinates at random."""
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

    def _get_pixels_avoiding_labels(self, avoid_labels):
        # Can only do this if an annotation is available
        if self.polygons is None:
            raise ValueError('No annotation is available.')

        # Generate annotated thumbnail with exact dimensions of tissue mask
        rows, cols = self.tissue_mask.shape
        _, mask, _ = self.get_thumbnail_with_annotation(
            size=(cols, rows), polygon_type='area')

        # Locate pixels in tissue and not in "avoid_labels"
        new_mask = np.where(self.tissue_mask == 1, 1, mask)
        new_mask = np.where(np.isin(mask, avoid_labels), 2, new_mask)

        return tuple(zip(*np.where(new_mask == 1)))

    def _max_repeated_pixel_ratio(self, image):
        """Compute ratio of count of most common pixel value to total count."""
        image = np.array(image)
        _, counts = np.unique(image, return_counts=True)

        return np.max(counts) / image.size

    def _get_area_ratio(self, mask, target_class):
        """Compute ratio of pixels labeled as 'target_class' to all pixels."""
        class_pixels = mask[np.where(mask == target_class)].size

        return class_pixels / mask.size

    def read_random_tissue_patch(self, level, size, avoid_labels=None):
        """Crop random patch from detected tissue on WSI.

        Parameters
        ----------
        level: int
            Slide level. Namesake argument to OpenSlide's "read_region"
            function.
        size: 2-tuple
            Crop size (width, height). Namesake argument to OpenSlide's
            "read_region" function.
        avoid_labels: iterable
            Labels from provided annotation to avoid when targeting tissue.
            An annotation is must be available.

        Returns
        -------
        Image region or patch as PIL RGBA image (plus annotation mask as 2D
        Numpy array if labels to avoid are provided).

        """
        if self.tissue_mask is None:
            self.get_tissue_mask()

        if avoid_labels is None:
            # Select coordinates matching tissue (labeled as 1)
            pixels_in_roi = tuple(zip(*np.where(self.tissue_mask == 1)))
        else:
            # Select coordinates matching tissue (labeled as 1) and avoiding
            # indicated labels
            # Can only do this if an annotation is available
            if self.polygons is None:
                raise ValueError('No annotation is available.')

            # Generate annotated thumbnail with exact dimensions of tissue mask
            rows, cols = self.tissue_mask.shape
            _, mask, _ = self.get_thumbnail_with_annotation(
                size=(cols, rows), polygon_type='area')

            # Locate pixels in tissue and not in "avoid_labels"
            new_mask = np.where(self.tissue_mask == 1, 1, mask)
            new_mask = np.where(np.isin(mask, avoid_labels), 2, new_mask)

            pixels_in_roi = tuple(zip(*np.where(new_mask == 1)))

        coordinates = self._get_random_coordinates(
            pixels_in_roi, level, self.downsampling_factor, size)
        patch = self.read_region(coordinates, level, size)

        # Avoid false positive regions where a large proportion of pixels is
        # exactly the same
        while self._max_repeated_pixel_ratio(patch) > 0.5:
            coordinates = self._get_random_coordinates(
                pixels_in_roi, level, self.downsampling_factor, size)
            patch = self.read_region(coordinates, level, size)

        # Also, make sure the patch is mostly normal tissue
        # (or backgound, not actually excluding it here...)
        if avoid_labels is not None:
            patch, mask = self.read_region_with_annotation(
                coordinates, level, size, polygon_type='area')
            avoid_ratios = [self._get_area_ratio(mask, x)
                            for x in avoid_labels]
            i = 0
            while any(x > 0.1 for x in avoid_ratios):
                i += 1
                coordinates = self._get_random_coordinates(
                    pixels_in_roi, level, self.downsampling_factor, size)
                patch, mask = self.read_region_with_annotation(
                    coordinates, level, size, polygon_type='area')
                avoid_ratios = [self._get_area_ratio(mask, x)
                                for x in avoid_labels]
                if i > 15:
                    raise ValueError(
                        'Cannot seem to find patch matching requirements.')

            return patch, mask

        return patch

    def _pick_random_coordinates(self, level, size, target_class):
        """Pick random patch to crop from WSI."""
        # Get thumbnail and select coordinates at random
        _, mask, downsampling_factor = self.get_thumbnail_with_annotation(
             size=(5000, 5000), polygon_type='area')

        # Select coordinates matching target class
        pixels_in_roi = tuple(zip(*np.where(mask == target_class)))

        coordinates = self._get_random_coordinates(
            pixels_in_roi, level, downsampling_factor, size)

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
        Image region or patch as PIL RGBA image plus annotation mask as 2D
        Numpy array.

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
