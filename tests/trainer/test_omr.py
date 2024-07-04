"""Test the BboxPairing class from the OMR module."""

import unittest
from pathlib import Path

import numpy as np

from konfuzio_sdk.trainer.omr import BboxPairing


class TestBboxPairing(unittest.TestCase):
    def setUp(self) -> None:
        self.bbox_pairing = BboxPairing()
        # for debug purposes, the bounding boxes can be visualized with BboxPairing.plot_bbox_pairs
        self.class1_boxes = [(2, 2, 3, 3), (3, 6, 9, 8), (10, 12, 12, 14), (10, 9, 12, 11)]
        self.class2_boxes = [(5, 2, 6, 3), (13, 13, 15, 15), (4, 10, 6, 12), (13, 10, 15, 12)]

    def test_mid_points(self) -> None:
        """
        Test the calculation of middle points of the bounding boxes edges.
        """
        # middle points of a 2x2 square
        boxes = [(0, 0, 2, 2)]
        expected_output = np.array([[[1, 0], [1, 2], [0, 1], [2, 1]]])
        np.testing.assert_array_equal(self.bbox_pairing._mid_points(boxes), expected_output)

    def test_pair_distances(self) -> None:
        """
        Test the calculation of distances between pairs of points from a 1x1 square.
        """
        # point pairs of a square 1x1 with the diagonal been ~1.41
        points1 = np.array([[0, 0], [0, 1]])
        points2 = np.array([[1, 0], [1, 1]])
        expected_output = np.array([[1.0, 1.41421356], [1.41421356, 1.0]])
        np.testing.assert_array_almost_equal(self.bbox_pairing._pair_distances(points1, points2), expected_output)

    def test_min_edge_distances(self) -> None:
        """
        Test the calculation of the minimum edge-to-edge distances between boxes in two classes.
        """
        class1_boxes = [(0, 0, 2, 2)]
        class2_boxes = [(3, 0, 5, 2)]
        expected_output = np.array([[1.0]])
        np.testing.assert_array_equal(
            self.bbox_pairing._min_edge_distances(class1_boxes, class2_boxes), expected_output
        )

    def test_find_pairs(self) -> None:
        """
        Test the optimal pairing of boxes from two classes.
        """
        class1_ind, class2_ind = self.bbox_pairing.find_pairs(self.class1_boxes, self.class2_boxes)
        expected_class1_ind = np.array([0, 1, 2, 3])
        expected_class2_ind = np.array([0, 2, 1, 3])
        np.testing.assert_array_equal(class1_ind, expected_class1_ind)
        np.testing.assert_array_equal(class2_ind, expected_class2_ind)

    def test_plot_pairs(self) -> None:
        """
        Test the plotting of the optimal pairing of boxes from two classes.
        """
        save_path = 'test.png'
        class1_boxes = self.class1_boxes  # blue
        class2_boxes = self.class2_boxes  # red
        class1_ind, class2_ind = self.bbox_pairing.find_pairs(class1_boxes, class2_boxes)
        self.bbox_pairing.plot_pairs(class1_boxes, class2_boxes, class1_ind, class2_ind, save_path=save_path)
        assert Path(save_path).exists()
        Path(save_path).unlink()
