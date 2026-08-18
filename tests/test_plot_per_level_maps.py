import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neup_inference_of_flow.field_reconstruction.plots import plot_per_level_maps


class PlotPerLevelMapsTests(unittest.TestCase):
    def _visible_axes_count(self, axs):
        flat_axes = np.atleast_1d(axs).ravel()
        return sum(ax.get_visible() for ax in flat_axes)

    def test_supported_level_counts_have_expected_visible_axes(self):
        for levels in (1, 2, 4, 5, 9):
            with self.subTest(levels=levels):
                maps = np.random.rand(levels, 4, 4)
                fig, axs = plot_per_level_maps(maps, titles=[f"T{i + 1}" for i in range(levels)])
                self.assertEqual(self._visible_axes_count(axs), levels)
                plt.close(fig)

    def test_short_or_missing_titles_do_not_raise(self):
        maps = np.random.rand(5, 3, 3)

        fig, _ = plot_per_level_maps(maps, titles=["Only one"])  # short titles list
        plt.close(fig)

        fig, _ = plot_per_level_maps(maps, titles=None)  # fallback titles
        plt.close(fig)

    def test_invalid_level_counts_raise_value_error(self):
        for levels in (0, 10):
            with self.subTest(levels=levels):
                maps = np.random.rand(levels, 2, 2)
                with self.assertRaises(ValueError):
                    plot_per_level_maps(maps)


if __name__ == '__main__':
    unittest.main()
