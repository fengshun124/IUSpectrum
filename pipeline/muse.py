import warnings
from typing import Union

import numpy as np
from astropy import constants
from astropy.io import fits
from joblib import delayed
from ppxf import ppxf_util
from ppxf.ppxf import ppxf
from ppxf.sps_util import sps_lib

from utils.parallel import ParallelTqdm

# get speed of light from astropy in km/s
SPEED_OF_LIGHT = constants.c.to('km/s').value


class MUSECube:
    def __init__(
            self,
            filename: str,
            redshift: float,
            wvl_air_angstrom_range: tuple[float, float] = (4822, 5212),
    ) -> None:
        """
        Read MUSE data cube, extract relevant information and preprocess it
        (de-redshift, ppxf log-rebin, and form coordinate grid).
        @param filename: filename of the MUSE data cube
        @param redshift: redshift of the galaxy
        @param wvl_air_angstrom_range: wavelength range to consider (Angstrom in air wavelength)
        """
        self._filename = filename
        self._wvl_air_angstrom_range = wvl_air_angstrom_range
        self._redshift = redshift

        self._read_fits_file()
        self._preprocess_cube()

        # initialize fields for storing results
        self._optimal_tmpls = None

    def _read_fits_file(self) -> None:
        """
        Read MUSE cube data, create a dummy noise cube, extract the wavelength axis,
        and obtain instrumental information from the FITS header.
        """
        cut_lhs, cut_rhs = 1, 1

        with fits.open(self._filename) as fits_hdu:
            # load fits header info
            self._fits_hdu_header = fits_hdu[0].header
            # read the cube data and apply scaling
            self._raw_cube_data = fits_hdu[0].data[cut_lhs:-cut_rhs, :, :] * 1e18
            # create a variance cube (dummy, as the file contains no errors)
            self._raw_cube_var = np.ones_like(self._raw_cube_data)

            # calculate wavelength axis
            self._obs_wvl_air_angstrom = (
                    self._fits_hdu_header['CRVAL3'] + self._fits_hdu_header['CD3_3'] *
                    (np.arange(self._raw_cube_data.shape[0]) + cut_lhs)
            )

            self._FWHM_gal = 1
            # instrument specific parameters from ESO
            self._pxl_size_x = abs(np.sqrt(
                self._fits_hdu_header['CD1_1'] ** 2 + self._fits_hdu_header['CD2_1'] ** 2
            )) * 3600
            self._pxl_size_y = abs(np.sqrt(
                self._fits_hdu_header['CD1_2'] ** 2 + self._fits_hdu_header['CD2_2'] ** 2
            )) * 3600

    def _preprocess_cube(self):
        wvl_air_angstrom = self._obs_wvl_air_angstrom / (1 + self._redshift)

        valid_mask = (
                (wvl_air_angstrom > self._wvl_air_angstrom_range[0]) &
                (wvl_air_angstrom < self._wvl_air_angstrom_range[1])
        )
        self._wvl_air_angstrom = wvl_air_angstrom[valid_mask]
        self._cube_data = self._raw_cube_data[valid_mask, :, :]
        self._cube_var = self._raw_cube_var[valid_mask, :, :]

        # derive signal and noise
        signal_2d = np.nanmedian(self._cube_data, axis=0)
        noise_2d = np.sqrt(np.nanmedian(self._cube_var, axis=0))
        self._signal = signal_2d.ravel()
        self._noise = noise_2d.ravel()

        # create spatial coordinates for each spaxel using the image indices
        ny, nx = signal_2d.shape  # note: cube shape is (n_wave, ny, nx)
        rows, cols = np.indices((ny, nx))
        # find the index of the brightest spaxel
        brightest_idx = np.argmax(signal_2d)
        # centering coordinates on the brightest spaxel and scale to arcseconds
        self.x = (cols.ravel() - cols.ravel()[brightest_idx]) * self._pxl_size_x
        self.y = (rows.ravel() - rows.ravel()[brightest_idx]) * self._pxl_size_y

        # reshape cube to 2D: each column corresponds to one spaxel spectrum
        n_wvl = self._cube_data.shape[0]
        self._spectra_2d = self._cube_data.reshape(n_wvl, -1)
        self._variance_2d = self._cube_var.reshape(n_wvl, -1)

        # log-rebin of the spectra
        self._vel_scale = np.min(
            SPEED_OF_LIGHT * np.diff(np.log(self._wvl_air_angstrom))
        )

        self._spectra, self._ln_lambda_gal, _ = ppxf_util.log_rebin(
            lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
            spec=self._spectra_2d,
            velscale=self._vel_scale
        )
        self._log_variance, _, _ = ppxf_util.log_rebin(
            lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
            spec=self._variance_2d,
            velscale=self._vel_scale
        )
        self._lambda_gal = np.exp(self._ln_lambda_gal)
        self._FWHM_gal = self._FWHM_gal / (1 + self._redshift)

        self._row = rows.ravel() + 1
        self._col = cols.ravel() + 1

        # initialize fields for storing results
        self._ny, self._nx = ny, nx
        self._velocity_field = np.empty((ny, nx))
        self._dispersion_field = np.empty((ny, nx))
        self._bestfit_field = np.empty((self._spectra.shape[0], ny, nx))
        self._apoly = []

    # first time fit (FTF)
    def fit_spectra(
            self,
            template_filename: str,
            ppxf_vel_init: int = 0,
            ppxf_vel_disp_init: int = 40,
            ppxf_deg: int = 3,
            n_jobs: int = -1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Fit the stellar continuum in each spaxel using pPXF.
        @param template_filename: Filename of the stellar template
        @param ppxf_vel_init: Initial guess for the velocity in pPXF
        @param ppxf_vel_disp_init: Initial guess for the velocity dispersion in pPXF
        @param ppxf_deg: Degree of the additive polynomial for pPXF
        @param n_jobs: Number of parallel jobs to run (-1 means using all processors).
        @return: Tuple of velocity field, dispersion field, bestfit field, optimal templates, and additive polynomials.
        """

        # load template
        sps = sps_lib(
            filename=template_filename,
            velscale=self._vel_scale,
            fwhm_gal=None,
            norm_range=self._wvl_air_angstrom_range
        )
        sps.templates = sps.templates.reshape(sps.templates.shape[0], -1)
        # normalize stellar template
        sps.templates /= np.median(sps.templates)
        tmpl_mask = ppxf_util.determine_mask(
            ln_lam=self._ln_lambda_gal,
            lam_range_temp=np.exp(sps.ln_lam_temp[[0, -1]]),
            width=1000
        )
        # update optimal templates shape
        self._optimal_tmpls = np.empty((sps.templates.shape[0], self._ny, self._nx))

        n_wvl, n_spaxel = self._spectra.shape

        def fit_spaxel(idx):
            i, j = np.unravel_index(idx, (self._ny, self._nx))
            galaxy_data = self._spectra[:, idx]
            # Use the square root of the variance as the noise estimate
            galaxy_noise = np.sqrt(self._log_variance[:, idx])

            if np.count_nonzero(galaxy_data) < 50:
                return i, j, None

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=RuntimeWarning,
                    message='invalid value encountered in scalar divide'
                )
                pp = ppxf(
                    sps.templates, galaxy_data, galaxy_noise,
                    self._vel_scale, mask=tmpl_mask,
                    start=[ppxf_vel_init, ppxf_vel_disp_init], degree=ppxf_deg,
                    lam=self._lambda_gal, lam_temp=sps.lam_temp,
                    quiet=True
                )
            return i, j, (
                pp.sol[0], pp.sol[1], pp.bestfit,
                sps.templates @ pp.weights,
                pp.apoly
            )

        fit_results = ParallelTqdm(
            n_jobs=n_jobs, desc='Fitting spectra', total_tasks=n_spaxel
        )(delayed(fit_spaxel)(idx) for idx in range(n_spaxel))
        for fit_result in fit_results:
            if fit_result[2] is None:
                continue
            row, col, (vel, disp, bestfit, optimal_tmpl, apoly_val) = fit_result
            self._velocity_field[row, col] = vel
            self._dispersion_field[row, col] = disp
            self._bestfit_field[:, row, col] = bestfit
            self._optimal_tmpls[:, row, col] = optimal_tmpl
            self._apoly.append(apoly_val)

        return (self._velocity_field, self._dispersion_field,
                self._bestfit_field, self._optimal_tmpls, self._apoly)

    # second time fit (STF)
    def fit_emission_lines(self):
        pass

    @property
    def redshift(self):
        return self._redshift

    @property
    def raw_data(self):
        return {
            'obs_wvl_air_angstrom': self._obs_wvl_air_angstrom,
            'raw_cube_data': self._raw_cube_data,
            'raw_cube_var': self._raw_cube_var,
        }

    @property
    def instrument_info(self):
        return {
            'CD1_1': self._fits_hdu_header['CD1_1'],
            'CD1_2': self._fits_hdu_header['CD1_2'],
            'CD2_1': self._fits_hdu_header['CD2_1'],
            'CD2_2': self._fits_hdu_header['CD2_2'],
            'CRVAL1': self._fits_hdu_header['CRVAL1'],
            'CRVAL2': self._fits_hdu_header['CRVAL2'],
        }

    @property
    def fits_hdu_header(self):
        return self._fits_hdu_header

    @property
    def fit_spectra_result(self):
        return {
            'velocity_field': self._velocity_field,
            'dispersion_field': self._dispersion_field,
            'bestfit_field': self._bestfit_field,
            'optimal_tmpls': self._optimal_tmpls,
            'apoly': self._apoly
        }

    def get_grid_stat(
            self, x_idx: Union[int, list[int]], y_idx: Union[int, list[int]]
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the velocity, dispersion, and bestfit for the given spaxel indices.
        @param x_idx: x index of the spaxel
        @param y_idx: y index of the spaxel
        @return: Dictionary of velocity, dispersion, and bestfit for the given spaxel indices.
        """
        if isinstance(x_idx, int):
            x_idx = [x_idx]
        if isinstance(y_idx, int):
            y_idx = [y_idx]

        return {
            (x, y): (
                self._velocity_field[y, x],
                self._dispersion_field[y, x],
                self._bestfit_field[:, y, x]
            )
            for x in x_idx for y in y_idx
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(filename={self._filename!r}, '
                f'redshift={self._redshift!r}, '
                f'wvl_air_angstrom_range={self._wvl_air_angstrom_range!r})')


if __name__ == '__main__':
    example_muse_cube = MUSECube(
        filename='../data/MUSE/VCC1588_stack.fits',
        redshift=.0042
    )

    example_muse_cube.fit_spectra(
        template_filename='../data/templates/spectra_emiles_9.0.npz'
    )

    print(
        example_muse_cube,
        example_muse_cube.instrument_info,
        example_muse_cube.raw_data,
        example_muse_cube.fit_spectra_result,
        sep='\n'
    )
