import time

import numpy as np
import pywt
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy import ndimage, signal
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def cwt_spectrogram(x, fs, nNotes=12, detrend=True, normalize=True):
    N = len(x)
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x = signal.detrend(x, type="linear")
    if normalize:
        stddev = x.std()
        x = x / stddev

    ###########################################################################
    # Define some parameters of our wavelet analysis.

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = int(np.log2(2 * np.floor(N / 2.0)))
    scales = 2 ** np.arange(1, nOctaves, 1.0 / nNotes)

    #     print (scales)

    ###########################################################################
    # cwt and the frequencies used.
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef, freqs = pywt.cwt(x, scales, "cmor1.5-1.0")
    frequencies = pywt.scale2frequency("cmor1.5-1.0", scales) / dt

    ###########################################################################
    # power
    #     power = np.abs(coef)**2
    power = np.abs(coef * np.conj(coef))

    # smooth a bit
    power = ndimage.gaussian_filter(power, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2 * np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4 * np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2)
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0 / coi

    return power, times, frequencies, coif


def spectrogram_plot(z, times, frequencies, coif, cmap=None, norm=Normalize(), ax=None, colorbar=True):
    ###########################################################################
    # plot

    # set default colormap, if none specified
    if cmap is None:
        cmap = get_cmap("Greys")
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    xx, yy = np.meshgrid(times, frequencies)
    ZZ = z

    im = ax.pcolor(xx, yy, ZZ, norm=norm, cmap=cmap)
    ax.plot(times, coif)
    ax.fill_between(times, coif, step="mid", alpha=0.4)

    # if colorbar:
    #     cbaxes = inset_axes(ax, width="2%", height="90%", loc=4)
    #     fig.colorbar(im, cax=cbaxes, orientation="vertical")

    # ax.set_xlim(times.min(), times.max())
    # ax.set_ylim(frequencies.min(), frequencies.max())

    return ax


def main():
    wav_data = np.load("standardized_sample.npy")
    sampling_frequency = 20

    n_samples = len(wav_data)
    total_duration = n_samples / sampling_frequency
    sample_times = np.linspace(0, total_duration, n_samples)
    plt.rcParams["figure.figsize"] = (16, 6)

    ###########################################################################
    # calculate spectrogram

    t0 = time.time()
    power, times, frequencies, coif = cwt_spectrogram(wav_data, sampling_frequency, nNotes=24)
    print(time.time() - t0)

    ###########################################################################
    # plot

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(sample_times, wav_data, color="b")

    ax1.set_xlim(0, total_duration)
    ax1.set_xlabel("time (s)")
    ax1.set_ylim(-np.abs(wav_data).max() * 1.2, np.abs(wav_data).max() * 1.2)
    ax1.grid(True)
    # ax1.axis('off')
    spectrogram_plot(power, times, frequencies, coif, cmap="jet", norm=LogNorm(), ax=ax2)

    ax2.set_xlim(0, total_duration)
    # ax2.set_ylim(0, 0.5*sampling_frequency)
    ax2.set_ylim(2.0 / total_duration, 0.5 * sampling_frequency)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")

    ax2.grid(True)
