#include <stdint.h>

/* Fixed arithmetic order, -ffp-contract=off, and no fast-math. */
double coarse80_residual(const double *theta, const uint64_t *masks, int mover_white) {
    double total = 0.0;
    for (int color = 0; color < 2; ++color) {
        const double sign = ((color == 1) == (mover_white != 0)) ? 1.0 : -1.0;
        for (int piece = 0; piece < 5; ++piece) {
            uint64_t bits = masks[color * 5 + piece];
            while (bits != 0) {
                const int square = __builtin_ctzll(bits);
                bits &= bits - UINT64_C(1);
                const int oriented = color == 1 ? square : (square ^ 56);
                const int rank = oriented >> 3;
                const int file = oriented & 7;
                const int zone = (rank >> 1) * 4 + (file >> 1);
                total += sign * theta[piece * 16 + zone];
            }
        }
    }
    return total;
}
