"""Physical orientation, Jacobian and rendering regressions for Wigner plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

sf = pytest.importorskip('strawberryfields')
from strawberryfields.backends.states import BaseFockState
from quannto.utils.strawberryfields.general_tools import coherent_ket
from quannto.utils.strawberryfields import wigner_plotters as wp


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    yield
    plt.close('all')


@pytest.mark.parametrize('hbar', [1., 2.])
@pytest.mark.parametrize('textbook', [False, True])
def test_complex_coherent_orientation_normalization_and_first_moments(monkeypatch, hbar, textbook):
    monkeypatch.setattr(sf, 'hbar', hbar)
    alpha = .7-.4j
    state = BaseFockState(coherent_ket(alpha, 24), 1, True, 24)
    W, x, p, _, _ = wp._compute_wigner_Z(
        state, 0, grid_lim=9, grid_pts=201, use_textbook_axes=textbook)
    effective_hbar = 1 if textbook else hbar
    x0, p0 = np.sqrt(2*effective_hbar)*np.array([alpha.real, alpha.imag])
    X, P = np.meshgrid(x, p)
    expected = np.exp(-((X-x0)**2+(P-p0)**2)/effective_hbar)/(np.pi*effective_hbar)
    np.testing.assert_allclose(W, expected, atol=1e-12, rtol=0)
    px, pp = np.trapz(W, p, axis=0), np.trapz(W, x, axis=1)
    assert np.trapz(px, x) == pytest.approx(1, abs=1e-8)
    assert np.trapz(pp, p) == pytest.approx(1, abs=1e-8)
    assert np.trapz(x*px, x) == pytest.approx(x0, abs=1e-8)
    assert np.trapz(p*pp, p) == pytest.approx(p0, abs=1e-8)


def test_sf_array_order_on_rectangular_grid(monkeypatch):
    monkeypatch.setattr(sf, 'hbar', 2.)
    state = BaseFockState(coherent_ket(1, 24), 1, True, 24)
    x, p = np.linspace(-4, 4, 81), np.linspace(-3, 3, 61)
    W = state.wigner(0, x, p)
    assert W.shape == (61, 81)
    row, col = np.unravel_index(W.argmax(), W.shape)
    assert (x[col], p[row]) == pytest.approx((2, 0))


def test_comparison_and_single_plot_share_density_and_full_grid(monkeypatch, tmp_path):
    monkeypatch.setattr(sf, 'hbar', 2.)
    monkeypatch.setattr(wp, 'figures_dir', lambda: tmp_path)
    state = BaseFockState(coherent_ket(.7-.4j, 24), 1, True, 24)
    kw = dict(grid_lim=5., grid_pts=81, use_textbook_axes=True)
    W, _, _, _, _ = wp._compute_wigner_Z(state, 0, **kw)
    wp.plot_wigner_2d_compare(state_L=state, state_R=state, **kw)
    for ax in plt.gcf().axes[:2]:
        np.testing.assert_allclose(ax.images[0].get_array(), W)
        assert ax.get_xlim() == (-5, 5)
        assert ax.get_ylim() == (-5, 5)
        assert ax.get_aspect() == 1
    wp.plot_wigner_2d(state=state, title='test', **kw)
    image = plt.gcf().axes[0].images[0]
    np.testing.assert_allclose(image.get_array(), W)
    assert image.norm.vmax == pytest.approx(W.max())
    assert image.norm.vmax > .175  # no old hbar=2 color clipping


def test_nonzero_mode_diagnostics_and_marginals(monkeypatch):
    monkeypatch.setattr(sf, 'hbar', 2.)
    ket = np.outer(coherent_ket(0, 20), coherent_ket(.6-.3j, 20))
    wp.plot_state_diagnostics(ket=ket, N=2, cutoff=20, mode=1,
                              grid_lim=6, grid_pts=121)
    axes = plt.gcf().axes
    for ax, mean in zip(axes, np.sqrt(2)*np.array([.6, -.3])):
        x, density = ax.lines[0].get_data()
        assert np.trapz(density, x) == pytest.approx(1, abs=1e-8)
        assert np.trapz(x*density, x) == pytest.approx(mean, abs=1e-8)


def test_3d_surface_uses_same_density(monkeypatch):
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    monkeypatch.setattr(sf, 'hbar', 2.)
    state = BaseFockState(coherent_ket(.7-.4j, 20), 1, True, 20)
    original = Axes3D.plot_surface
    captured = {}
    def surface(self, X, P, Z, **kwargs):
        captured['Z'] = Z.copy()
        return original(self, X, P, Z, **kwargs)
    monkeypatch.setattr(Axes3D, 'plot_surface', surface)
    W, _, _, _, _ = wp._compute_wigner_Z(state, 0, grid_lim=5, grid_pts=41,
                                        use_textbook_axes=True)
    wp.plot_wigner_3d(state=state, grid_lim=5, grid_pts=41,
                     downsample=2, add_projections=True)
    np.testing.assert_allclose(captured['Z'], W[::2, ::2])
