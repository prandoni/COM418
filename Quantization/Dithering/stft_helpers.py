# Functions to make the quantization notebooks less crowded with code.

# Author: Mihailo Kolundzija (mihailo.kolundzija@epfl.ch)
# Date:   May 20th, 2020

import numpy as np


def stft_window(fft_size=1024, block_size=1024, overlap=0.5,
                win_type='hanning', symmetric_zp=False):
    '''
    Computes the analysis STFT window.

    Parameters
    ----------
    fft_size : int, optional (default = 1024)
        Size of the STFT segment.
    block_size : int, optional (default = 1024)
        Size of the non-zero block inside the segment.
    overlap : float, optional (default = 0.5)
        Overlap between consecutive blocks.
    win_type : {'hanning', 'hamming', 'rect'}, optional (default = 'hanning')
        Type of the window used.
    symmetric_zp : bool, optional (default = False)
        If True, apply symmetric zero-padding; if False, zero-pad at the end.

    Returns
    -------
        win : 1D array_like
            (fft_size,) array with the window function.
    '''
    zp_length = fft_size - block_size
    if symmetric_zp:
        start = zp_length // 2
    else:
        start = 0
    indices = range(start, start+block_size)

    # Compute the analysis window.
    win = np.zeros(fft_size)
    gain = 2 * (1 - overlap)

    if win_type.lower() == 'hanning':
        win[indices] = np.hanning(block_size)
    elif win_type.lower() == 'hamming':
        win[indices] = np.hamming(block_size)
    elif win_type.lower() == 'rect':
        win[indices] = 1
    else:
        raise ValueError('Window type could be hanning, hamming or rect')
    return gain * win


def stft(x, win, hop_size, fullspec=False):
    '''
    Compute the STFT  of a signal.

    Parameters
    ----------
    x : array_like
        (L,) array with the signal.
    win : 1D array_like
        (W,) array with the window function used for STFT computation.
    hop_size : int
        Hop size between two consecutive segments.
    fullspec : bool, optional (default = False)
        Computing full spectrum or only positive frequencies.

    Returns
    -------
        seg_offsets : array_like
        (n_segments,) array the index of the first sample of every segment.
        freqs : array_like
        (spec_size,) array with bin frequencies normalized by f_s/2 (Nyquist).
        sxy : array_like
        (n_segments, spec_size) STFT of the cross-spectral density.
    '''

    # Compute the analysis window.
    fft_size = len(win)

    if fullspec:
        spec_size = fft_size
    else:
        # Use only with frequencies up to the Nyquist.
        spec_size = fft_size//2+1 if fft_size % 2 == 0 else (fft_size+1)//2

    freqs = np.arange(0, spec_size) / fft_size

    # Get the starts of the segments to be processed.
    seg_offsets = np.arange(0, x.shape[0]-fft_size, hop_size)
    n_segments = len(seg_offsets)

    S = np.zeros((spec_size, n_segments), dtype=np.complex128)
    n = 0
    for offset in seg_offsets:
        if fullspec:
            S[:, n] = np.fft.fft(win * x[offset:offset+fft_size])
        else:
            S[:, n] = np.fft.rfft(win * x[offset:offset+fft_size])
        n += 1
    return seg_offsets, freqs, S
