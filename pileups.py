import numpy as np




def tad_metric(pileup):
    assert pileup.shape == (100,100)

    inner_slice = (slice(27,47), slice(53,73))
    forward_slice = (slice(51,71), slice(77,97))
    backward_slice = (slice(3,23), slice(29,49))

    mid = np.nanmean(pileup[inner_slice])
    top = np.nanmean(pileup[forward_slice])
    bot = np.nanmean(pileup[backward_slice])

    out = (top+bot)/2

    return mid/out


def dot_metric(pileup):
    assert pileup.shape[0] == pileup.shape[1]
    assert pileup.shape[0] >= 20

    N = pileup.shape[0]

    w = int(N/20)

    m_peak = int(N/2)
    peak_slice = slice(m_peak - w, m_peak + w), slice(m_peak - w, m_peak + w)

    m_bg1 = int(2*N/10)
    bg_slice1 = (slice(m_bg1 - w, m_bg1 + w), slice(m_bg1 - w, m_bg1 + w))

    m_bg2 = int(8*N/10)
    bg_slice2 = (slice(m_bg2 - w, m_bg2 + w), slice(m_bg2 - w, m_bg2 + w))

    peak = np.nanmean(pileup[peak_slice])
    bg1 = np.nanmean(pileup[bg_slice1])
    bg2 = np.nanmean(pileup[bg_slice2])

    bg = (bg1 + bg2)/2

    return peak/bg
