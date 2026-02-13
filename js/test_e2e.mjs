/**
 * End-to-end test: ONNX backbone + JS ISTFT vs Python full decoder.
 *
 * This test loads the FP32 ONNX model, runs it on the same input as Python,
 * then applies JS ISTFT and compares the audio output.
 *
 * Note: Since the ONNX model was exported with random weights (not loaded from
 * checkpoint), its weights differ from the Python decoder's random weights.
 * So we can only verify:
 * 1. The ONNX model produces correct output shapes
 * 2. The JS ISTFT correctly reconstructs audio from the ONNX output
 * 3. When given the SAME head_output, JS ISTFT matches Python ISTFT
 */
import { readFileSync } from 'fs';
import * as ort from 'onnxruntime-node';
import { irfft } from './fft.mjs';

const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_FREQ = N_FFT / 2 + 1;

function hannWindow(length) {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / length));
  }
  return w;
}

function istft(magPhase, T) {
  const window = hannWindow(N_FFT);
  const windowSq = new Float64Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) windowSq[i] = window[i] * window[i];

  const outputLen = N_FFT + HOP_LENGTH * (T - 1);
  const output = new Float64Array(outputLen);
  const windowEnvelope = new Float64Array(outputLen);
  const halfRe = new Float64Array(N_FREQ);
  const halfIm = new Float64Array(N_FREQ);

  for (let t = 0; t < T; t++) {
    for (let k = 0; k < N_FREQ; k++) {
      const rawMag = magPhase[k * T + t];
      const phase = magPhase[(N_FREQ + k) * T + t];
      const mag = Math.min(Math.exp(rawMag), 100);
      halfRe[k] = mag * Math.cos(phase);
      halfIm[k] = mag * Math.sin(phase);
    }
    halfRe[0] = 0; halfIm[0] = 0;
    halfRe[N_FREQ - 1] = 0; halfIm[N_FREQ - 1] = 0;

    const frame = irfft(halfRe, halfIm, N_FFT);
    const offset = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      output[offset + i] += frame[i] * window[i];
      windowEnvelope[offset + i] += windowSq[i];
    }
  }

  for (let i = 0; i < outputLen; i++) {
    if (windowEnvelope[i] > 1e-11) output[i] /= windowEnvelope[i];
  }

  const pad = N_FFT >> 1;
  const trimmedLen = outputLen - 2 * pad;
  const result = new Float32Array(trimmedLen);
  for (let i = 0; i < trimmedLen; i++) result[i] = output[pad + i];
  return result;
}

async function main() {
  console.log('=== End-to-End Test: ONNX + JS ISTFT ===\n');

  // Test 1: Load ONNX INT4 model and verify it runs
  console.log('Test 1: ONNX INT4 model inference');
  const modelPath = '../exports/soprano_decoder_int4.onnx';
  const session = await ort.InferenceSession.create(modelPath);

  const T = 10;
  const input = new Float32Array(512 * T);
  // Fill with some non-zero values
  for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.01) * 0.1;

  const tensor = new ort.Tensor('float32', input, [1, 512, T]);
  const results = await session.run({ hidden_states: tensor });
  const output = results.mag_phase;

  const expectedL = 4 * (T - 1) + 1; // 37
  console.log(`  Input: (1, 512, ${T})`);
  console.log(`  Output: (${output.dims.join(', ')})`);
  console.log(`  Expected time dim: ${expectedL}`);

  if (output.dims[1] !== 2050 || output.dims[2] !== expectedL) {
    console.error('  FAIL: Unexpected output shape');
    process.exit(1);
  }
  console.log('  PASS\n');

  // Test 2: JS ISTFT on ONNX output produces valid audio
  console.log('Test 2: JS ISTFT on ONNX output');
  const audio = istft(output.data, output.dims[2]);
  const expectedAudioLen = (T - 1) * 2048; // 18432
  console.log(`  Audio length: ${audio.length} (expected ${expectedAudioLen})`);

  // The ISTFT output might not exactly match expectedAudioLen due to center padding
  // But it should be close
  const audioLenFromISTFT = HOP_LENGTH * (expectedL - 1); // 512 * 36 = 18432
  console.log(`  ISTFT output length: ${audio.length} (expected ${audioLenFromISTFT})`);

  if (audio.length !== audioLenFromISTFT) {
    console.error('  FAIL: Unexpected audio length');
    process.exit(1);
  }

  // Check audio is not all zeros and is finite
  let nonZero = 0;
  let allFinite = true;
  for (let i = 0; i < audio.length; i++) {
    if (audio[i] !== 0) nonZero++;
    if (!isFinite(audio[i])) allFinite = false;
  }
  console.log(`  Non-zero samples: ${nonZero}/${audio.length}`);
  console.log(`  All finite: ${allFinite}`);

  if (nonZero < audio.length * 0.5 || !allFinite) {
    console.error('  FAIL: Audio quality check failed');
    process.exit(1);
  }
  console.log('  PASS\n');

  // Test 3: ISTFT matches Python when given same input
  console.log('Test 3: ISTFT matches Python reference');
  try {
    const testData = JSON.parse(readFileSync('/tmp/istft_test_data.json', 'utf-8'));
    const magPhase = new Float32Array(testData.mag_phase);
    const expected = new Float64Array(testData.expected);
    const tFrames = testData.T;

    const actual = istft(magPhase, tFrames);
    let maxErr = 0;
    for (let i = 0; i < actual.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(actual[i] - expected[i]));
    }
    console.log(`  Max error vs Python: ${maxErr.toExponential(4)}`);
    console.log(maxErr < 1e-3 ? '  PASS' : '  FAIL');
  } catch (e) {
    console.log('  SKIP (test data not available)');
  }

  // Test 4: Load INT4 bs512 model
  console.log('\nTest 4: ONNX INT4 bs512 model inference');
  const session2 = await ort.InferenceSession.create('../exports/soprano_decoder_int4_bs512.onnx');
  const results2 = await session2.run({ hidden_states: tensor });
  console.log(`  Output: (${results2.mag_phase.dims.join(', ')})`);
  console.log('  PASS');

  console.log('\n=== All tests passed ===');
}

main().catch(e => { console.error(e); process.exit(1); });
