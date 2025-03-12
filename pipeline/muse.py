import numpy as np
from astropy import constants
from astropy.io import fits
from ppxf import ppxf_util

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
        (de-redshift, ppxf log-rebin, form coordinate grid).
        @param filename: filename of the MUSE data cube
        @param redshift: redshift of the galaxy
        @param wvl_air_angstrom_range: wavelength range to consider (Angstrom in air wavelength)
        """
        self._filename = filename
        self._wvl_air_angstrom_range = wvl_air_angstrom_range
        self._redshift = redshift

        self._read_fits_file()
        self._preprocess_cube()

    def _read_fits_file(self) -> None:
        """
        Read MUSE cube data, create a dummy noise cube, extract the wavelength axis,
        and obtain instrumental information from the FITS header.
        """
        cut_lhs = 1
        cut_rhs = 1

        with fits.open(self._filename) as fits_hdu:
            # load fits header info
            self._fits_hdu_header = fits_hdu[0].header
            # read the cube data and apply scaling
            self._raw_cube_data = fits_hdu[0].data[cut_lhs:-cut_rhs, :, :] * 1e18
            # create a variance cube (dummy, as the file contains no errors)
            self._raw_cube_var = np.empty_like(self._raw_cube_data)

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

        log_wvl = np.log(self._wvl_air_angstrom)
        delta_log_wvl = np.diff(log_wvl)
        vel_scale_init = SPEED_OF_LIGHT * np.min(delta_log_wvl)

        self._spectra, self._ln_lambda_gal, self._vel_scale = ppxf_util.log_rebin(
            lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
            spec=self._spectra_2d,
            velscale=vel_scale_init
        )
        self._FWHM_gal = self._FWHM_gal / (1 + self._redshift)

        self._row = rows.ravel() + 1
        self._col = cols.ravel() + 1

        self._velocity_field = np.empty((ny, nx))
        self._signal_field = np.empty((ny, nx))

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

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(filename={self._filename!r}, '
                f'redshift={self._redshift!r}, '
                f'wvl_air_angstrom_range={self._wvl_air_angstrom_range!r})')


if __name__ == '__main__':
    example_muse_cube = MUSECube(
        filename='../data/MUSE/VCC0308_stack.fits',
        redshift=.0042
    )

    print(
        example_muse_cube,
        example_muse_cube.instrument_info,
        example_muse_cube.raw_data,
        sep='\n'
    )
