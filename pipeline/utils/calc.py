from typing import Union, Optional

import numpy as np


def apply_velocity_shift(
        wvl: Union[list[float], np.ndarray],
        z: float,
) -> np.ndarray:
    """
    Apply velocity shift to the given wavelength.
    :param wvl: original wavelength
    :param z: redshift in km/s
    :return: shifted wavelength
    """
    return np.ndarray(np.asarray(wvl) * (1 + z))


# re-implement of SpectRes (https://github.com/ACCarnall/SpectRes/)
def _calc_wvl_bin(
        wvl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the bin edges and widths of given wavelength array."""
    midpoints = .5 * (wvl[1:] + wvl[:-1])

    edges = np.concatenate([
        np.array([wvl[0] - 0.5 * (wvl[1] - wvl[0])]),
        midpoints,
        np.array([wvl[-1] + 0.5 * (wvl[-1] - wvl[-2])])
    ])

    widths = np.diff(edges)

    return edges, widths


def resample_spectrum(
        new_wvl: Union[list[float], np.ndarray],
        src_wvl: Union[list[float], np.ndarray],
        src_flux: Union[list[float], np.ndarray],
        src_flux_err: Optional[Union[list[float], np.ndarray]] = None,
        fill: float = 0.0,
) -> Union[
    tuple[np.ndarray, np.ndarray], np.ndarray
]:
    # convert input to numpy array
    new_wvl = np.asarray(new_wvl)
    src_wvl = np.asarray(src_wvl)
    src_flux = np.asarray(src_flux)

    if src_wvl.shape != src_flux.shape:
        raise ValueError(
            f'src_wvl and src_flux must have the same shape. '
            f'Got {src_wvl.shape} and {src_flux.shape}'
        )

    if src_flux_err is not None:
        src_flux_err = np.asarray(src_flux_err)
        if src_flux_err.shape != src_flux.shape:
            raise ValueError(
                f'src_flux_err must have the same shape as src_flux. '
                f'Got {src_flux_err.shape} and {src_flux.shape}'
            )

    # compute bin edges for source and new wavelength grids
    src_edges, src_widths = _calc_wvl_bin(src_wvl)
    new_edges, _ = _calc_wvl_bin(new_wvl)

    new_flux = np.full(new_wvl.shape, fill, dtype=src_flux.dtype)
    if src_flux_err is not None:
        new_flux_err = np.full(new_wvl.shape, fill, dtype=src_flux.dtype)
    else:
        new_flux_err = None

    for i in range(len(new_wvl)):
        if new_edges[i] < src_edges[0] or new_edges[i + 1] > src_edges[-1]:
            continue

        # identify the indices of the source bins overlapping with the new bin
        start_idx = np.searchsorted(src_edges, new_edges[i], side='right') - 1
        stop_idx = np.searchsorted(src_edges, new_edges[i + 1], side='left') - 1

        # if the new bin is fully contained within a single source bin
        if start_idx == stop_idx:
            new_flux[i] = src_flux[start_idx]
            if new_flux_err is not None:
                new_flux_err[i] = src_flux_err[start_idx]
            continue

        # for multiple overlapping bins, adjust the first and last bin contributions
        partial_widths = src_widths[start_idx:stop_idx + 1].copy()
        # fraction of the last source bin that overlaps the new bin
        start_factor = (
                (src_edges[start_idx + 1] - new_edges[i]) /
                (src_edges[start_idx + 1] - src_edges[start_idx])
        )
        partial_widths[0] *= start_factor
        end_factor = (
                (new_edges[i + 1] - src_edges[stop_idx]) /
                (src_edges[stop_idx + 1] - src_edges[stop_idx])
        )
        partial_widths[-1] *= end_factor

        # calculate weighted flux for the new bin
        flux_slice = src_flux[start_idx:stop_idx + 1]
        total_width = np.sum(partial_widths)
        new_flux[i] = np.sum(flux_slice * partial_widths) / total_width

        # propagate uncertainties if provided
        if new_flux_err is not None:
            err_slice = src_flux_err[start_idx:stop_idx + 1]
            weighted_err_sq = np.sum((err_slice * partial_widths) ** 2)
            new_flux_err[i] = np.sqrt(weighted_err_sq) / total_width

    if new_flux_err is not None:
        return new_flux, new_flux_err
    return new_flux
