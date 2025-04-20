import numpy as np


def k_set_generator(period, f):  #period
    """
        k_set_generator : generator a set of selected k
        period: by discrete Fourier transform, the sampling interval [-period/2, period/2] lead to the period or
        f : the number of selected frequency points
    """
    # Define the original signal function
    def original_signal(x, y, z):
        """
        Original signal function: a Gaussian-like signal.
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.exp(-5 * r)

    # Parameters
    N = 64  # Number of samples in each dimension (Nyquist rule)  #N*N*N

    # Create spatial coordinates
    min_val, max_val = -period / 2, period / 2
    x = np.linspace(min_val, max_val, N)
    y = np.linspace(min_val, max_val, N)
    z = np.linspace(min_val, max_val, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the original signal
    original_signal_3d = original_signal(X, Y, Z)

    # Perform 3D FFT
    fft_result = np.fft.fftn(original_signal_3d)
    fft_shifted = np.fft.fftshift(fft_result)

    # Compute frequency coordinates
    kx = np.fft.ifftshift(np.fft.fftfreq(N, x[1] - x[0]))
    ky = np.fft.ifftshift(np.fft.fftfreq(N, y[1] - y[0]))
    kz = np.fft.ifftshift(np.fft.fftfreq(N, z[1] - z[0]))
    L = kx[1] - kx[0]  # Frequency spacing
    L1 = X[1, 0, 0] - X[0, 0, 0]  # Spatial spacing



    # Select a limited number of frequency points
    num_selected_freq = f  # Number of selected frequency points
    selected_indices = np.argsort(np.abs(fft_shifted.flatten()))[-num_selected_freq:]  # Select the largest amplitude points

    selected_indices_unraveled = np.unravel_index(selected_indices, fft_shifted.shape)
    fft_selected = np.zeros_like(fft_shifted, dtype=complex)
    fft_selected[selected_indices_unraveled] = fft_shifted[selected_indices_unraveled]

    k_m = np.zeros((num_selected_freq, 3))
    f_m_r = np.zeros((num_selected_freq,))
    f_m_i = np.zeros((num_selected_freq,))
    for i in range(num_selected_freq):
        ix = int(selected_indices_unraveled[0][i])
        iy = int(selected_indices_unraveled[1][i])
        iz = int(selected_indices_unraveled[2][i])
        fx = kx[ix]
        fy = ky[iy]
        fz = kz[iz]
        GK = fft_shifted[ix][iy][iz]
        k_m[i][0] = fx * 2 * np.pi / N / L / L1
        k_m[i][1] = fy * 2 * np.pi / N / L / L1
        k_m[i][2] = fz * 2 * np.pi / N / L / L1
        f_m_r[i] = GK.real
        f_m_i[i] = GK.imag
    np.savez('S'+ str(int(period))+'-'+ str(f) +'.npz',k=k_m,fr=f_m_r,fi = f_m_i)

k_set_generator(12, 512)