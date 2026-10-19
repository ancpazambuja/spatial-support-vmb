import numpy as np
import math
import os



def simulate_convolution_trf(prev_trf, k, s=1, p=None):

    if p is None:
        p = (k - 1) // 2

    l = len(prev_trf)

    prev_trf_padded = np.pad(
        prev_trf,
        ((p, p), (p, p), (0, 0), (0, 0)),
        mode='edge'
    )

    m = (l - k + 2*p) // s + 1

    trf = np.empty((m, m, 2, 2), dtype=int)
    #print(trf)

    max_trf_size = 0

    for i in range(m):
        for j in range(m):

            top_left = prev_trf_padded[i*s, j*s]
            bottom_right = prev_trf_padded[i*s+k-1, j*s+k-1]

            trf_size_a = (
                bottom_right[1, 0]
                - top_left[0, 0]
                + 1
            )

            trf_size_b = (
                bottom_right[1, 1]
                - top_left[0, 1]
                + 1
            )

            trf_size = int(
                math.sqrt(trf_size_a * trf_size_b)
            )

            max_trf_size = max(
                max_trf_size,
                trf_size
            )

            trf[i, j, 0] = top_left[0]
            trf[i, j, 1] = bottom_right[1]

    return trf, max_trf_size


def simulate_maxpool_trf(prev_trf):

    k = 2

    out_shape = prev_trf.shape[0] // k
    print(f"out_shape={out_shape}")

    trf = np.zeros(
        (out_shape, out_shape, 2, 2),
        dtype=int
    )

    max_trf_size = 0

    for i in range(out_shape):
        for j in range(out_shape):

            top_left = prev_trf[i*k, j*k]
            bottom_right = prev_trf[i*k+k-1, j*k+k-1]

            trf[i, j, 0] = top_left[0]
            trf[i, j, 1] = bottom_right[1]

            trf_size_a = (
                bottom_right[1, 0]
                - top_left[0, 0]
                + 1
            )

            trf_size_b = (
                bottom_right[1, 1]
                - top_left[0, 1]
                + 1
            )

            trf_size = int(
                math.sqrt(trf_size_a * trf_size_b)
            )

            max_trf_size = max(
                max_trf_size,
                trf_size
            )

    return trf, max_trf_size


def simulate_deconv_trf(prev_trf):

    k = 2
    s = 2

    in_dim = prev_trf.shape[0]
    print(f"in_dim = {in_dim}")

    out_dim = (in_dim - 1) * s + k

    trf = np.full(
        (out_dim, out_dim, 2, 2),
        -1,
        dtype=int
    )

    max_trf_size = 0

    for i in range(in_dim):
        for j in range(in_dim):

            t = j*s
            b = j*s + k - 1

            l = i*s
            r = i*s + k - 1

            for m in range(l, r+1):
                for n in range(t, b+1):

                    if np.array_equal(
                        trf[m, n],
                        np.array([[-1, -1], [-1, -1]])
                    ):
                        trf[m, n] = prev_trf[i, j]

                    else:

                        trf[m, n, 0, 0] = min(
                            trf[m, n, 0, 0],
                            prev_trf[i, j, 0, 0]
                        )

                        trf[m, n, 0, 1] = min(
                            trf[m, n, 0, 1],
                            prev_trf[i, j, 0, 1]
                        )

                        trf[m, n, 1, 0] = max(
                            trf[m, n, 1, 0],
                            prev_trf[i, j, 1, 0]
                        )

                        trf[m, n, 1, 1] = max(
                            trf[m, n, 1, 1],
                            prev_trf[i, j, 1, 1]
                        )

                    trf_size_a = (
                        trf[m, n, 1, 0]
                        - trf[m, n, 0, 0]
                        + 1
                    )

                    trf_size_b = (
                        trf[m, n, 1, 1]
                        - trf[m, n, 0, 1]
                        + 1
                    )

                    trf_size = int(
                        math.sqrt(trf_size_a * trf_size_b)
                    )

                    max_trf_size = max(
                        max_trf_size,
                        trf_size
                    )

    return trf, max_trf_size


def concatenate_trf(decoder_trf, skip_trf):

    out = np.zeros_like(decoder_trf)

    out[:, :, 0, 0] = np.minimum(
        decoder_trf[:, :, 0, 0],
        skip_trf[:, :, 0, 0]
    )

    out[:, :, 0, 1] = np.minimum(
        decoder_trf[:, :, 0, 1],
        skip_trf[:, :, 0, 1]
    )

    out[:, :, 1, 0] = np.maximum(
        decoder_trf[:, :, 1, 0],
        skip_trf[:, :, 1, 0]
    )

    out[:, :, 1, 1] = np.maximum(
        decoder_trf[:, :, 1, 1],
        skip_trf[:, :, 1, 1]
    )

    return out


def create_input_trf(input_dim):

    return np.array(
        [
            [
                [[[j, i], [j, i]]
                 for i in range(input_dim)]
                for j in range(input_dim)
            ]
        ][0],
        dtype=int
    )


def compute_unet_trf(
        input_dim=1024,
        depth=5,
        kernel_size=3):

    print("[ INFO ] Creating input...")
    trf = create_input_trf(input_dim)
    print("[ DONE ] ")

    skips = []

    # encoder
    for dd in range(depth):

        print(f"depth = {dd}")
        trf, trf_size = simulate_convolution_trf(
            trf,
            kernel_size
        )
        print(f"trf_size = {trf_size}")

        trf, trf_size = simulate_convolution_trf(
            trf,
            kernel_size
        )
        print(f"trf_size = {trf_size}")

        skips.append(trf.copy())

        trf, _ = simulate_maxpool_trf(trf)

    # bottleneck
    trf, _ = simulate_convolution_trf(
        trf,
        kernel_size
    )

    trf, _ = simulate_convolution_trf(
        trf,
        kernel_size
    )

    # decoder
    for level in reversed(range(depth)):

        trf, _ = simulate_deconv_trf(trf)

        skip_trf = skips[level]

        H = min(
            trf.shape[0],
            skip_trf.shape[0]
        )

        trf = concatenate_trf(
            trf[:H, :H],
            skip_trf[:H, :H]
        )

        trf, _ = simulate_convolution_trf(
            trf,
            kernel_size
        )

        trf, _ = simulate_convolution_trf(
            trf,
            kernel_size
        )

    center = trf.shape[0] // 2

    rect = trf[center, center]

    h = (
        rect[1, 0]
        - rect[0, 0]
        + 1
    )

    w = (
        rect[1, 1]
        - rect[0, 1]
        + 1
    )

    trf_size = int(
        math.sqrt(h * w)
    )

    return trf_size, rect


if __name__ == "__main__":
    k = 3
    #for d in [2,3,4,5]:
    d=5
    #for k in [3, 5, 7, 9, 13, 17, 19, 25, 33, 37, 49]:
    for k in [3, 5]:

        trf_size, rect = compute_unet_trf(
            input_dim=1024,
            depth=d,
            kernel_size=k
        )

        print(
            f"-----> [ RESULT ] k={k:2d}  d={d:2d}  TRF={trf_size}"
        )