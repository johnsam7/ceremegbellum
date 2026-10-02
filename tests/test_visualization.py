import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import cmb.visualization as cmb_viz


@pytest.fixture
def fake_pyvista(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Replace pyvista and pyvistaqt with mocks and pretend a display is available."""
    pv = MagicMock(name="pyvista")
    pvqt = MagicMock(name="pyvistaqt")
    monkeypatch.setitem(sys.modules, "pyvista", pv)
    monkeypatch.setitem(sys.modules, "pyvistaqt", pvqt)
    monkeypatch.setattr(cmb_viz, "_OFFSCREEN", False)
    return pv, pvqt


def _make_src() -> tuple[dict, np.ndarray]:
    """Make a tiny single-triangle source space and data for it."""
    src = {
        "rr": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "tris": np.array([[0, 1, 2]]),
    }
    return src, np.array([0.0, 0.5, 1.0])


def _make_geo() -> tuple[dict, np.ndarray]:
    """Make a tiny cerebellum geometry dict for plot_inflated."""
    src, data = _make_src()
    geo = {
        "verts_inflated_fs": src["rr"],
        "dw_data": {"sparse": np.array([0, 1, 2]), "sparse_tris": src["tris"]},
    }
    return geo, data


def test_plot_normal_pyvista_backend(fake_pyvista) -> None:
    """Default backend uses pyvista.Plotter and calls show()."""
    pv, pvqt = fake_pyvista
    src, data = _make_src()

    plotter = cmb_viz.plot_normal(src, data, show=True, offscreen=False)

    pv.Plotter.assert_called_once_with(
        window_size=[1200, 1200], off_screen=False, notebook=False
    )
    pvqt.BackgroundPlotter.assert_not_called()
    assert plotter is pv.Plotter.return_value
    pv.Plotter.return_value.show.assert_called_once()


@pytest.mark.parametrize("show", [True, False])
def test_plot_normal_pyvistaqt_backend(fake_pyvista, show: bool) -> None:
    """The pyvistaqt backend uses BackgroundPlotter and never calls blocking show()."""
    pv, pvqt = fake_pyvista
    src, data = _make_src()

    plotter = cmb_viz.plot_normal(
        src, data, src_cortex=src, show=show, backend="pyvistaqt"
    )

    pvqt.BackgroundPlotter.assert_called_once_with(window_size=(1200, 1200), show=show)
    pv.Plotter.assert_not_called()
    assert plotter is pvqt.BackgroundPlotter.return_value
    pvqt.BackgroundPlotter.return_value.show.assert_not_called()
    assert pvqt.BackgroundPlotter.return_value.add_mesh.call_count == 2


def test_plot_inflated_pyvistaqt_screenshot(fake_pyvista, tmp_path) -> None:
    """The pyvistaqt backend renders before taking a screenshot."""
    _, pvqt = fake_pyvista
    geo, data = _make_geo()
    fname = str(tmp_path / "shot.png")

    plotter = cmb_viz.plot_inflated(
        geo, data, "sparse", show=False, screenshot_fname=fname, backend="pyvistaqt"
    )

    assert plotter is pvqt.BackgroundPlotter.return_value
    pvqt.BackgroundPlotter.return_value.render.assert_called_once()
    pvqt.BackgroundPlotter.return_value.screenshot.assert_called_once_with(fname)


@pytest.mark.parametrize(
    "kwargs",
    [{"notebook_inline": True}, {"offscreen": True}],
    ids=["notebook_inline", "offscreen"],
)
def test_pyvistaqt_incompatible_options(fake_pyvista, kwargs: dict) -> None:
    """The pyvistaqt backend rejects notebook and offscreen rendering."""
    src, data = _make_src()
    with pytest.raises(ValueError, match="not supported with backend='pyvistaqt'"):
        cmb_viz.plot_normal(src, data, backend="pyvistaqt", **kwargs)


def test_pyvistaqt_headless_raises(fake_pyvista, monkeypatch) -> None:
    """The pyvistaqt backend raises when no display is available."""
    monkeypatch.setattr(cmb_viz, "_OFFSCREEN", True)
    geo, data = _make_geo()
    with pytest.raises(RuntimeError, match="requires a display"):
        cmb_viz.plot_inflated(geo, data, "sparse", backend="pyvistaqt")


def test_pyvistaqt_missing_raises(fake_pyvista, monkeypatch) -> None:
    """A helpful ImportError is raised when pyvistaqt cannot be imported."""
    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "pyvistaqt", None)
    src, data = _make_src()
    with pytest.raises(ImportError, match="requires pyvistaqt and a Qt binding"):
        cmb_viz.plot_normal(src, data, backend="pyvistaqt")


def test_unknown_backend_raises(fake_pyvista) -> None:
    """An unknown backend name raises a ValueError."""
    src, data = _make_src()
    with pytest.raises(ValueError, match="Unknown backend"):
        cmb_viz.plot_normal(src, data, backend="matplotlib")  # pyright: ignore[reportArgumentType]
