/**
 * Radix-2 Cooley-Tukey FFT for power-of-2 sizes.
 * Used by the ISTFT to convert frequency-domain frames back to time-domain audio.
 */

/**
 * Reverse bits of an integer.
 * @param {number} x - Input integer
 * @param {number} bits - Number of bits
 * @returns {number} Bit-reversed integer
 */
function bitReverse(x, bits) {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

/**
 * In-place iterative radix-2 FFT.
 * @param {Float64Array} re - Real parts (modified in place)
 * @param {Float64Array} im - Imaginary parts (modified in place)
 * @param {boolean} inverse - If true, compute IFFT
 */
export function fft(re, im, inverse = false) {
  const N = re.length;
  if (N & (N - 1)) throw new Error(`FFT size must be power of 2, got ${N}`);

  // Bit-reversal permutation
  const bits = Math.log2(N);
  for (let i = 0; i < N; i++) {
    const j = bitReverse(i, bits);
    if (j > i) {
      let tmp;
      tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // Cooley-Tukey butterfly
  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size >> 1;
    const angle = sign * 2 * Math.PI / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < N; i += size) {
      let tRe = 1, tIm = 0;
      for (let j = 0; j < halfSize; j++) {
        const a = i + j;
        const b = a + halfSize;

        const uRe = re[a], uIm = im[a];
        const vRe = re[b] * tRe - im[b] * tIm;
        const vIm = re[b] * tIm + im[b] * tRe;

        re[a] = uRe + vRe;
        im[a] = uIm + vIm;
        re[b] = uRe - vRe;
        im[b] = uIm - vIm;

        const newTRe = tRe * wRe - tIm * wIm;
        tIm = tRe * wIm + tIm * wRe;
        tRe = newTRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < N; i++) {
      re[i] /= N;
      im[i] /= N;
    }
  }
}

/**
 * Inverse Real FFT: complex half-spectrum -> real signal.
 *
 * Takes the first N/2+1 complex bins (the non-redundant half of a
 * conjugate-symmetric spectrum) and reconstructs N real samples.
 *
 * @param {Float64Array} halfRe - Real parts of spectrum, length N/2+1
 * @param {Float64Array} halfIm - Imaginary parts of spectrum, length N/2+1
 * @param {number} N - Output signal length (must be power of 2)
 * @returns {Float64Array} Real-valued time-domain signal of length N
 */
export function irfft(halfRe, halfIm, N) {
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const halfN = (N >> 1) + 1;

  // Copy non-redundant bins
  for (let i = 0; i < halfN; i++) {
    re[i] = halfRe[i];
    im[i] = halfIm[i];
  }

  // Reconstruct conjugate-symmetric second half
  for (let i = 1; i < N >> 1; i++) {
    re[N - i] = halfRe[i];
    im[N - i] = -halfIm[i];
  }

  fft(re, im, true);
  return re; // Imaginary part is ~zero due to symmetry
}
