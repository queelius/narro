/**
 * Test .soprano file parsing.
 * Run: node test_npz.mjs
 */
import { readFileSync } from 'fs';
import * as fflate from 'fflate';
import { parseSopranoFile } from './soprano-decoder.mjs';

const buffer = readFileSync('/tmp/test_decode.soprano.npz');
const encoded = parseSopranoFile(buffer.buffer, fflate);

console.log('=== NPZ Parser Test ===');
console.log(`Model ID: ${encoded.modelId}`);
console.log(`Format version: ${encoded.formatVersion}`);
console.log(`Sample rate: ${encoded.sampleRate}`);
console.log(`Sentences: ${encoded.sentences.length}`);

const s = encoded.sentences[0];
console.log(`\nSentence 0:`);
console.log(`  Text: "${s.text}"`);
console.log(`  T: ${s.T}`);
console.log(`  Hidden states length: ${s.hiddenStates.length} (expected ${s.T * 512})`);
console.log(`  Token IDs: [${Array.from(s.tokenIds).join(', ')}]`);
console.log(`  Finish reason: ${s.finishReason}`);
console.log(`  Text index: ${s.textIndex}, Sentence index: ${s.sentenceIndex}`);

// Verify values
let pass = true;
if (s.T !== 5) { console.error('FAIL: T !== 5'); pass = false; }
if (s.hiddenStates.length !== 5 * 512) { console.error('FAIL: hiddenStates wrong size'); pass = false; }
if (s.tokenIds[0] !== 100) { console.error('FAIL: tokenIds[0] !== 100'); pass = false; }
if (s.text !== 'Hello world') { console.error('FAIL: text mismatch'); pass = false; }
if (encoded.modelId !== 'ekwek/Soprano-1.1-80M') { console.error('FAIL: modelId mismatch'); pass = false; }

console.log(pass ? '\nPASS' : '\nFAIL');
if (!pass) process.exit(1);
