/**
 * Full decode pipeline test: .soprano file -> ONNX -> ISTFT -> audio
 * Run: node test_full_decode.mjs
 */
import { readFileSync } from 'fs';
import * as ort from 'onnxruntime-node';
import * as fflate from 'fflate';
import { SopranoDecoder } from './soprano-decoder.mjs';

async function main() {
  console.log('=== Full Decode Pipeline Test ===\n');

  const decoder = new SopranoDecoder();

  // Load ONNX model
  console.log('Loading INT4 ONNX model...');
  const modelBuf = readFileSync('../exports/soprano_decoder_int4.onnx');
  await decoder.loadModel(modelBuf.buffer, ort);
  console.log('  Model loaded\n');

  // Load .soprano file
  console.log('Loading .soprano file...');
  const sopranoBuf = readFileSync('/tmp/test_decode.soprano.npz');
  const encoded = await decoder.loadSopranoFile(sopranoBuf.buffer, fflate);
  console.log(`  ${encoded.sentences.length} sentence(s), "${encoded.sentences[0].text}"\n`);

  // Decode
  console.log('Decoding...');
  const t0 = performance.now();
  const audio = await decoder.decode(encoded, (i, total) => {
    console.log(`  Sentence ${i + 1}/${total}`);
  });
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`\nDecoded in ${elapsed}s`);
  console.log(`  Audio length: ${audio.length} samples (${(audio.length / 32000).toFixed(2)}s)`);

  // Verify audio quality
  let nonZero = 0;
  let maxAbs = 0;
  let allFinite = true;
  for (let i = 0; i < audio.length; i++) {
    if (audio[i] !== 0) nonZero++;
    const abs = Math.abs(audio[i]);
    if (abs > maxAbs) maxAbs = abs;
    if (!isFinite(audio[i])) allFinite = false;
  }
  console.log(`  Non-zero: ${nonZero}/${audio.length}`);
  console.log(`  Max abs: ${maxAbs.toFixed(4)}`);
  console.log(`  All finite: ${allFinite}`);

  // Test WAV blob
  const blob = decoder.toWavBlob(audio);
  console.log(`  WAV size: ${(blob.size / 1024).toFixed(1)} KB`);

  const pass = audio.length > 0 && allFinite && nonZero > audio.length * 0.1;
  console.log(pass ? '\nPASS' : '\nFAIL');
  if (!pass) process.exit(1);
}

main().catch(e => { console.error(e); process.exit(1); });
