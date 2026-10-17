"""Artificial content hashes prevent duplicate-video leakage across evaluation groups."""
import unittest
from ml.baseline.core import split_videos


class ContentSplitTests(unittest.TestCase):
    def row(self, name, angle, digest):
        return {'video_id': name, 'angle_id': angle, 'label': 0,
                'content_sha256': digest}

    def test_identical_content_cannot_cross_train_holdout_even_with_different_names(self):
        rows = [self.row('fixture-one.mov', 'A', 'a' * 64),
                self.row('fixture-copy.mov', 'B', 'a' * 64)]
        with self.assertRaises(ValueError):
            split_videos(rows, ['A'], ['B'])

    def test_identical_content_can_stay_in_one_partition(self):
        rows = [self.row('fixture-one.mov', 'A', 'a' * 64),
                self.row('fixture-copy.mov', 'A', 'a' * 64),
                self.row('fixture-other.mov', 'B', 'b' * 64)]
        result = split_videos(rows, ['A'], ['B'])
        self.assertEqual(len(result['train']), 2)
        self.assertEqual(len(result['holdout']), 1)

    def test_present_content_hashes_must_be_complete_and_valid(self):
        for bad in [None, '', 'not-a-sha256', ['not-hashable']]:
            rows = [self.row('fixture-one.mov', 'A', 'a' * 64),
                    self.row('fixture-other.mov', 'B', bad)]
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    split_videos(rows, ['A'], ['B'])
        rows = [self.row('fixture-one.mov', 'A', 'a' * 64),
                {'video_id': 'fixture-other.mov', 'angle_id': 'B', 'label': 0}]
        with self.assertRaises(ValueError):
            split_videos(rows, ['A'], ['B'])


if __name__ == '__main__':
    unittest.main()
