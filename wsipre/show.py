# -*- coding: utf-8 -*-

"""
Show images, such as thumbnails or regions extracted from whole-slide images
(WSI; also known as virtual slides).

"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatch


class Figure(object):
    """An open *annotated* WSI.

    Parameters
    ----------
    image: PIL Image
        Image to display; a thumbnail or region read from WSI.
    annotation: Numpy 2D array
        Annotation labels for each pixel in the input ``image``.
    color_map: dict {label int: RGB color list}
        Mapping of RGB color values to class labels. Colors are triplet lists
        of the R, G, and B values.

    Attributes
    ----------
    image: PIL Image
        The input ``image``.
    annotation: Numpy 2D array
        The input ``annotation``.
    color_map: dict {label: color}
        Mapping between each label and the color for visualization.

    Examples
    --------
    >>> from wsipre import show
    >>> colors = {0: (0, 0, 0), 2: (1, 0, 0)}
    >>> fig = show.Figure(image=image, annotation=mask, color_map=colors)
    >>> fig.color_map
    {0: (0, 0, 0), 2: (1, 0, 0)}

    >>> fig.show_label_colors(height=5)

    >>> fig.show_image_with_annotation()

    """

    def __init__(self, image, annotation, color_map={0: (0., 0., 0.),
                                                     1: (0., .8, 1.),
                                                     2: (1., .8, 0.),
                                                     3: (1., .4, 0.)}):
        self.image = image
        self.annotation = annotation.astype(np.float32)  # In case it's boolean
        self.color_map = color_map

        # RGBA values should be within 0-1 range
        if all([0 <= channel <= 255 for color in self.color_map.values()
                for channel in color]):
            if any([channel > 1 for color in self.color_map.values()
                    for channel in color]):
                self.color_map = {key: [channel / 255.0 for channel in color]
                                  for (key, color) in self.color_map.items()}
        else:
            raise ValueError(
                'Please make sure color triplets values in "color_map" are ' +
                'either between 0-1 or 0-255 range.')

        # Are there enough colors?
        labels = np.unique(self.annotation)
        labels_not_in_map = [label not in color_map.keys() for label in labels]
        if any(labels_not_in_map):
            raise ValueError(
                'No color provided in "color_map" for label(s) ' +
                f' {list(labels[labels_not_in_map])}.')

    def show_label_colors(self, width=20, height=6, font_color='w',
                          font_size=14):
        """Display a color bar with the colors and overlayed labels.

        Parameters
        ----------
        width: int
            Width of each class rectangle in plot.
        height: int
            Height of each class rectangle in plot.
        font_size: int
            Size of the font used to display class labels.

        Returns
        -------
        Matplotlib image illustrating mapping between class label and color.

        """
        rectangles = [mpatch.Rectangle((0 + width * i, 0), width, height,
                                       color=list(self.color_map.values())[i])
                      for i in range(len(self.color_map.keys()))]
        rectangles = {list(self.color_map.keys())[i]: rect
                      for i, rect in enumerate(rectangles)}

        fig, ax = plt.subplots()

        for r in rectangles:
            ax.add_artist(rectangles[r])
            rx, ry = rectangles[r].get_xy()
            cx = rx + rectangles[r].get_width() / 2.0
            cy = ry + rectangles[r].get_height() / 2.0

            ax.annotate(r, (cx, cy), color=font_color, weight='bold',
                        fontsize=font_size, ha='center', va='center')

        ax.set_xlim((0, width * len(rectangles)))
        ax.set_ylim((0, height))
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()

    def show_image(self):
        """Show loaded image.

        Display the loaded WSI thumbnail or region.

        Returns
        -------
        Matplotlib image.

        """
        plt.imshow(self.image)
        plt.show()

    def _paint_annotation_mask(self):
        """Convert 2D mask to 3D format (RGB image)."""
        red = self.annotation.copy()
        green = self.annotation.copy()
        blue = self.annotation.copy()

        for label, color in self.color_map.items():
            idx = self.annotation == label
            red[idx], green[idx], blue[idx] = color

        rgb = np.stack([red, green, blue], axis=2)

        return rgb

    def show_annotation(self):
        """Show loaded image annotation.

        Display the loaded WSI thumbnail or region annotation, corresponding to
        a mask of pixel-wise labels.

        Returns
        -------
        Matplotlib image.

        """
        rgb_mask = self._paint_annotation_mask()
        plt.imshow(rgb_mask)
        plt.show()

    def show_image_with_annotation(self, split=True):
        """Show loaded image and annotation.

        Parameters
        ----------
        split: bool
            Whether to display the split image and annotation side-by-side or a
            combined visualization of the annotation on the image.

        Returns
        -------
        Matplotlib image.

        """
        if split:
            rgb_mask = self._paint_annotation_mask()

            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.imshow(np.array(self.image))
            ax2.imshow(rgb_mask)
            plt.show()
        else:
            image = np.array(self.image)[:, :, :3]
            if np.max(image) > 1:
                image = image / 255.

            for label, color in self.color_map.items():
                if label == 0:
                    continue
                idx = self.annotation == label
                image[idx] = color

            plt.imshow(image)
            plt.show()
