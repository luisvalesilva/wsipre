# -*- coding: utf-8 -*-

"""
wsipre.annotation_reader
~~~~~~~~~~~~~~~~~~~~~~~~

Functionality to read whole-slide image (WSI) annotations (from XML files).
"""

from xml.dom import minidom


def asap_annotations(xml_file):
    """Get annotations from XML file in the ASAP style.

    Collects region annotations bounded by polygonal contours from an XML
    annotation file in the ASAP style (popularized by the CAMELYON challenge),
    along with their labels.

    Parameters
    ----------
    xml_file: str
        Path to XML file.
    Returns
    -------
    List of regions (lists of 2-tuples of x, y polygon vertex coordinates) and
    list of region labels.
    """
    xml = minidom.parse(xml_file)
    annotations = xml.getElementsByTagName('Annotation')

    polygons, labels = [], []
    for annotation in annotations:
        label = annotation.attributes['PartOfGroup'].value
        if label == 'None':
            continue

        labels.append(label)
        vertices = annotation.getElementsByTagName('Coordinate')

        polygon = []
        for vertex in vertices:
            polygon.append((float(vertex.attributes['X'].value),
                            float(vertex.attributes['Y'].value)))
        polygons.append(polygon)

    return polygons, labels


def bach_annotations(xml_file):
    """Get annotations in XML file from the BACH ICIAR 2018 challenge.

    Collects region annotations bounded by polygonal contours from an XML
    annotation file, along with their labels.

    Parameters
    ----------
    xml_file: str
        Path to XML file.
    Returns
    -------
    List of regions (lists of 2-tuples of x, y polygon vertex coordinates) and
    list of region labels.
    """
    xml = minidom.parse(xml_file)
    annotations = xml.getElementsByTagName('Region')

    polygons, labels = [], []
    for annotation in annotations:
        vertices = annotation.getElementsByTagName('Vertex')
        label = annotation.getElementsByTagName('Attribute')
        if label:
            labels.append(label[0].attributes['Value'].value)
        else:
            labels.append(annotation.getAttribute('Text'))

        polygon = []
        for vertex in vertices:
            polygon.append((float(vertex.attributes['X'].value),
                            float(vertex.attributes['Y'].value)))
        polygons.append(polygon)

    return polygons, labels
