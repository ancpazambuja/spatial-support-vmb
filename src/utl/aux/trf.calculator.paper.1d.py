
import numpy as np
import math

def debug(msg, allow=False):
    if allow:
        print(msg)

def simulate_convolution_trf(prev_trf, k, s=1, p=None):
    if p is None:
        p = (k - 1) // 2

    l = len(prev_trf)

    # 1D Padding along the single spatial axis
    prev_trf_padded = np.pad(
        prev_trf,
        ((p, p), (0, 0)),
        mode='edge'
    )

    m = (l - k + 2*p) // s + 1
    trf = np.empty((m, 2), dtype=int)
    max_trf_size = 0

    for i in range(m):
        # 1D Boundary checks
        left = prev_trf_padded[i*s]
        right = prev_trf_padded[i*s+k-1]

        trf_size = right[1] - left[0] + 1
        max_trf_size = max(max_trf_size, trf_size)

        trf[i, 0] = left[0]
        trf[i, 1] = right[1]

    return trf, max_trf_size


def simulate_maxpool_trf(prev_trf):
    k = 2
    out_shape = prev_trf.shape[0] // k

    trf = np.zeros((out_shape, 2), dtype=int)
    max_trf_size = 0

    for i in range(out_shape):
        left = prev_trf[i*k]
        right = prev_trf[i*k+k-1]

        trf[i, 0] = left[0]
        trf[i, 1] = right[1]

        trf_size = right[1] - left[0] + 1
        max_trf_size = max(max_trf_size, trf_size)

    return trf, max_trf_size


def simulate_deconv_trf(prev_trf):
    k = 2
    s = 2

    in_dim = prev_trf.shape[0]
    out_dim = (in_dim - 1) * s + k

    trf = np.full((out_dim, 2), -1, dtype=int)
    max_trf_size = 0

    for i in range(in_dim):
        l = i*s
        r = i*s + k - 1

        for m in range(l, r+1):
            if np.array_equal(trf[m], np.array([-1, -1])):
                trf[m] = prev_trf[i]
            else:
                trf[m, 0] = min(trf[m, 0], prev_trf[i, 0])
                trf[m, 1] = max(trf[m, 1], prev_trf[i, 1])

            trf_size = trf[m, 1] - trf[m, 0] + 1
            max_trf_size = max(max_trf_size, trf_size)

    return trf, max_trf_size


def concatenate_trf(decoder_trf, skip_trf):
    out = np.zeros_like(decoder_trf)
    out[:, 0] = np.minimum(decoder_trf[:, 0], skip_trf[:, 0])
    out[:, 1] = np.maximum(decoder_trf[:, 1], skip_trf[:, 1])
    return out


def create_input_trf(input_dim):
    # Generates standard index positions [[0,0], [1,1], ..., [N,N]] for 1D Tracking
    return np.array([[i, i] for i in range(input_dim)], dtype=int)


def compute_unet_trf(input_dim=1024, depth=5, kernel_size=3):
    debug("[ INFO ] Creating input...")
    trf = create_input_trf(input_dim)
    debug("[ DONE ] ")

    skips = []

    # encoder
    for dd in range(depth):
        debug(f"depth = {dd}")
        trf, trf_size = simulate_convolution_trf(trf, kernel_size)
        debug(f"pos conv -- trf_size = {trf_size}")

        trf, trf_size = simulate_convolution_trf(trf, kernel_size)
        debug(f"pos conv -- trf_size = {trf_size}")

        skips.append(trf.copy())
        trf, trf_size = simulate_maxpool_trf(trf)
        debug(f"pos maxpool -- trf_size = {trf_size}")

    # bottleneck
    trf, trf_size = simulate_convolution_trf(trf, kernel_size)
    debug(f"pos conv -- trf_size = {trf_size}")
    trf, trf_size = simulate_convolution_trf(trf, kernel_size)
    debug(f"pos conv -- trf_size = {trf_size}")

    # decoder
    for level in reversed(range(depth)):
        debug(f"depth = {level}")
        trf, trf_size = simulate_deconv_trf(trf)
        debug(f"pos deconv -- trf_size = {trf_size}")
        skip_trf = skips[level]

        H = min(trf.shape[0], skip_trf.shape[0])
        trf = concatenate_trf(trf[:H], skip_trf[:H])
        ##debug(f"pos concat -- trf_size = {trf_size}")

        trf, trf_size = simulate_convolution_trf(trf, kernel_size)
        debug(f"pos conv -- trf_size = {trf_size}")
        trf, trf_size = simulate_convolution_trf(trf, kernel_size)
        debug(f"pos conv -- trf_size = {trf_size}")

    center = trf.shape[0] // 2
    line = trf[center]
    trf_size = line[1] - line[0] + 1

    return trf_size, line


if __name__ == "__main__":
    input_dim=int(math.pow(2,14))
    debug(f"input_dim = {input_dim}", allow=True)
    d = 5
    for d in (2,3,4,5,6,7,8):
        print("----------------------------------------")
        for k in (3,5,7,9,11,13,17,19,25,33,37,49):
            trf_size, line = compute_unet_trf(
            input_dim=input_dim,
            depth=d,
            kernel_size=k
            )
            debug(f"-----> [ RESULT ] k={k:2d}  d={d:2d}  TRF={trf_size} Bounds={line.tolist()}", allow=True)
        #debug(f"k={k:2d}  r={1}  TRF={trf_size}", allow=True)




