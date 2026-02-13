/**
 * Soprano TTS Browser Decoder
 *
 * Decodes .soprano files (pre-encoded speech) to audio in the browser.
 * Uses onnxruntime-web for the neural network backbone and a pure-JS
 * ISTFT for the final audio reconstruction.
 *
 * Pipeline: .soprano file -> hidden states -> ONNX backbone -> mag/phase -> ISTFT -> PCM audio
 *
 * Usage:
 *   import { SopranoDecoder } from './soprano-decoder.mjs';
 *
 *   const decoder = new SopranoDecoder();
 *   await decoder.loadModel('/models/soprano_decoder_int4.onnx');
 *   const encoded = await decoder.loadSopranoFile('/audio/post.soprano');
 *   const audio = await decoder.decode(encoded);
 *   decoder.play(audio);
 */

import { irfft } from './fft.mjs';

// ---------------------------------------------------------------------------
// Constants (must match Python soprano.tts)
// ---------------------------------------------------------------------------

const SAMPLE_RATE = 32000;
const TOKEN_SIZE = 2048;       // Audio samples per decoder token
const HIDDEN_DIM = 512;        // LLM hidden state dimension
const N_FFT = 2048;            // STFT FFT size
const HOP_LENGTH = 512;        // STFT hop length
const WIN_LENGTH = 2048;       // STFT window length
const N_FREQ = N_FFT / 2 + 1; // 1025 frequency bins
const UPSCALE = 4;             // Decoder time upscale factor

// ---------------------------------------------------------------------------
// Float16 conversion
// ---------------------------------------------------------------------------

/**
 * Convert a Uint16Array of float16 values to a Float32Array.
 */
function float16ToFloat32Array(uint16arr) {
  const out = new Float32Array(uint16arr.length);
  for (let i = 0; i < uint16arr.length; i++) {
    const h = uint16arr[i];
    const sign = (h >> 15) & 1;
    const exponent = (h >> 10) & 0x1F;
    const mantissa = h & 0x3FF;

    let value;
    if (exponent === 0) {
      value = (mantissa / 1024) * (2 ** -14); // Subnormal or zero
    } else if (exponent === 31) {
      value = mantissa === 0 ? Infinity : NaN;
    } else {
      value = (1 + mantissa / 1024) * (2 ** (exponent - 15));
    }

    out[i] = sign ? -value : value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// NPZ / .soprano file parser
// ---------------------------------------------------------------------------

/**
 * Parse a .npy buffer into a typed array + shape.
 * @param {ArrayBuffer} buffer - Raw .npy file content
 * @returns {{ data: TypedArray, shape: number[], dtype: string }}
 */
function parseNpy(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Verify magic: \x93NUMPY
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4E) {
    throw new Error('Invalid .npy magic number');
  }

  const major = bytes[6];
  let headerLen, headerOffset;
  if (major >= 2) {
    headerLen = view.getUint32(8, true); // little-endian
    headerOffset = 12;
  } else {
    headerLen = view.getUint16(8, true);
    headerOffset = 10;
  }

  // Parse header string (Python dict literal)
  const headerStr = new TextDecoder().decode(bytes.slice(headerOffset, headerOffset + headerLen));
  const dataOffset = headerOffset + headerLen;

  // Extract dtype
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const dtype = descrMatch ? descrMatch[1] : '<f4';

  // Extract shape
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  let shape = [];
  if (shapeMatch && shapeMatch[1].trim()) {
    shape = shapeMatch[1].split(',').filter(s => s.trim()).map(s => parseInt(s.trim()));
  }

  // Read data based on dtype
  const rawBuf = buffer.slice(dataOffset);
  let data;

  switch (dtype) {
    case '<f2': { // float16 -> convert to float32
      const uint16 = new Uint16Array(rawBuf);
      data = float16ToFloat32Array(uint16);
      break;
    }
    case '<f4':
      data = new Float32Array(rawBuf);
      break;
    case '<f8':
      data = new Float64Array(rawBuf);
      break;
    case '<u2':
      data = new Uint16Array(rawBuf);
      break;
    case '<i4':
      data = new Int32Array(rawBuf);
      break;
    case '|u1':
      data = new Uint8Array(rawBuf);
      break;
    default:
      throw new Error(`Unsupported numpy dtype: ${dtype}`);
  }

  return { data, shape, dtype };
}

/**
 * Parse an .npz (ZIP of .npy) file.
 * Uses fflate for decompression.
 * @param {ArrayBuffer} buffer - Raw .npz file content
 * @param {object} fflate - The fflate module (must provide unzipSync)
 * @returns {Map<string, { data: TypedArray, shape: number[], dtype: string }>}
 */
function parseNpz(buffer, fflate) {
  const files = fflate.unzipSync(new Uint8Array(buffer));
  const result = new Map();
  for (const [filename, content] of Object.entries(files)) {
    const name = filename.replace(/\.npy$/, '');
    result.set(name, parseNpy(content.buffer));
  }
  return result;
}

/**
 * Parse a .soprano file into structured data.
 * @param {ArrayBuffer} buffer - Raw .soprano file content
 * @param {object} fflate - The fflate module
 * @returns {object} Parsed encoded speech with sentences and metadata
 */
export function parseSopranoFile(buffer, fflate) {
  const arrays = parseNpz(buffer, fflate);

  // Parse metadata JSON
  const metaArray = arrays.get('meta');
  const metaJson = new TextDecoder().decode(metaArray.data);
  const meta = JSON.parse(metaJson);

  // Build sentence objects
  const sentences = [];
  for (let i = 0; i < meta.num_sentences; i++) {
    const smeta = meta.sentences[i];
    const hiddenStates = arrays.get(`hidden_${i}`).data; // Float32Array, flat (T*512)
    const shape = arrays.get(`hidden_${i}`).shape;       // [T, 512]
    const T = shape[0];

    let attention = null;
    if (smeta.has_attention && arrays.has(`attention_${i}`)) {
      attention = arrays.get(`attention_${i}`).data;
    }

    sentences.push({
      hiddenStates,
      T,
      tokenIds: arrays.get(`token_ids_${i}`).data,
      tokenEntropy: arrays.get(`entropy_${i}`).data,
      finishReason: smeta.finish_reason,
      text: smeta.text,
      textIndex: smeta.text_index,
      sentenceIndex: smeta.sentence_index,
      attentionWeights: attention,
    });
  }

  return {
    sentences,
    modelId: meta.model_id,
    formatVersion: meta.format_version,
    sampleRate: meta.sample_rate,
    tokenAudioSamples: meta.token_audio_samples,
    hiddenDim: meta.hidden_dim,
    topP: meta.top_p,
    temperature: meta.temperature,
    repetitionPenalty: meta.repetition_penalty,
  };
}

// ---------------------------------------------------------------------------
// ISTFT (center padding mode)
// ---------------------------------------------------------------------------

/**
 * Generate a Hann window of given length.
 * @param {number} length
 * @returns {Float64Array}
 */
function hannWindow(length) {
  const w = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / length));
  }
  return w;
}

/**
 * Inverse STFT with center padding (matching PyTorch torch.istft center=True).
 *
 * The backbone outputs a (2050, T) tensor of [magnitude; phase]. This function:
 * 1. Splits into mag and phase (each 1025 x T)
 * 2. Applies exp() to magnitude, clips at 100
 * 3. Computes complex spectrum: mag * e^(j*phase)
 * 4. Zeros DC and Nyquist bins (as the Python ISTFTHead does for center mode)
 * 5. Performs overlap-add ISTFT with Hann window normalization
 * 6. Center-trims the output
 *
 * @param {Float32Array} magPhase - Backbone output, flat (2050 * T)
 * @param {number} T - Number of time frames
 * @returns {Float32Array} Audio signal
 */
function istft(magPhase, T) {
  const window = hannWindow(N_FFT);
  const windowSq = new Float64Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) windowSq[i] = window[i] * window[i];

  // Output buffer: full overlap-add length
  const outputLen = N_FFT + HOP_LENGTH * (T - 1);
  const output = new Float64Array(outputLen);
  const windowEnvelope = new Float64Array(outputLen);

  // Reusable buffers for IRFFT
  const halfRe = new Float64Array(N_FREQ);
  const halfIm = new Float64Array(N_FREQ);

  for (let t = 0; t < T; t++) {
    // Extract mag and phase for this frame, apply exp + clip, compute complex
    for (let k = 0; k < N_FREQ; k++) {
      const rawMag = magPhase[k * T + t];
      const phase = magPhase[(N_FREQ + k) * T + t];

      // exp(mag) clipped at 100 (matching Python: torch.exp(mag).clip(max=1e2))
      const mag = Math.min(Math.exp(rawMag), 100);

      halfRe[k] = mag * Math.cos(phase);
      halfIm[k] = mag * Math.sin(phase);
    }

    // Zero DC and Nyquist (Python: spec[:,0] = 0; spec[:,-1] = 0)
    halfRe[0] = 0; halfIm[0] = 0;
    halfRe[N_FREQ - 1] = 0; halfIm[N_FREQ - 1] = 0;

    // IRFFT: (1025 complex) -> (2048 real)
    const frame = irfft(halfRe, halfIm, N_FFT);

    // Window + overlap-add
    const offset = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      output[offset + i] += frame[i] * window[i];
      windowEnvelope[offset + i] += windowSq[i];
    }
  }

  // Normalize by window envelope
  for (let i = 0; i < outputLen; i++) {
    if (windowEnvelope[i] > 1e-11) {
      output[i] /= windowEnvelope[i];
    }
  }

  // Center trim: remove N_FFT/2 from each end
  const pad = N_FFT >> 1;
  const trimmedLen = outputLen - 2 * pad;
  const result = new Float32Array(trimmedLen);
  for (let i = 0; i < trimmedLen; i++) {
    result[i] = output[pad + i];
  }

  return result;
}

// ---------------------------------------------------------------------------
// Main decoder class
// ---------------------------------------------------------------------------

export class SopranoDecoder {
  constructor() {
    this.session = null;
    this.audioContext = null;
    this._fflate = null;
    this._ort = null;
  }

  /**
   * Load the ONNX decoder model.
   * @param {string|ArrayBuffer} modelSource - URL or ArrayBuffer of ONNX model
   * @param {object} ort - onnxruntime-web module
   * @param {object} [options] - ONNX session options
   */
  async loadModel(modelSource, ort, options = {}) {
    this._ort = ort;
    const sessionOptions = { ...options };

    // Default execution providers: wasm for browser, cpu for Node.js
    if (!sessionOptions.executionProviders) {
      sessionOptions.executionProviders =
        typeof window !== 'undefined' ? ['wasm'] : ['cpu'];
    }

    this.session = await ort.InferenceSession.create(modelSource, sessionOptions);
  }

  /**
   * Load and parse a .soprano file.
   * @param {string|ArrayBuffer} source - URL or ArrayBuffer
   * @param {object} fflate - fflate module
   * @returns {object} Parsed encoded speech
   */
  async loadSopranoFile(source, fflate) {
    this._fflate = fflate;
    let buffer;
    if (typeof source === 'string') {
      const response = await fetch(source);
      buffer = await response.arrayBuffer();
    } else {
      buffer = source;
    }
    return parseSopranoFile(buffer, fflate);
  }

  /**
   * Run the ONNX backbone on hidden states.
   * @param {Float32Array} hiddenStates - Flat array of shape (T, HIDDEN_DIM)
   * @param {number} T - Sequence length
   * @returns {Promise<{ data: Float32Array, T: number }>} Backbone output (mag_phase) and output time frames
   */
  async _runBackbone(hiddenStates, T) {
    if (!this.session) throw new Error('Model not loaded. Call loadModel() first.');

    // Transpose (T, 512) -> (1, 512, T) for ONNX input
    const input = new Float32Array(HIDDEN_DIM * T);
    for (let c = 0; c < HIDDEN_DIM; c++) {
      for (let t = 0; t < T; t++) {
        input[c * T + t] = hiddenStates[t * HIDDEN_DIM + c];
      }
    }

    const tensor = new this._ort.Tensor('float32', input, [1, HIDDEN_DIM, T]);
    const results = await this.session.run({ hidden_states: tensor });
    const output = results.mag_phase;

    // Output shape: [1, 2050, outT]
    const outT = output.dims[2];
    return { data: output.data, T: outT };
  }

  /**
   * Decode a single sentence's hidden states to audio.
   * @param {Float32Array} hiddenStates - Hidden states, flat (T, 512)
   * @param {number} T - Sequence length (tokens)
   * @returns {Promise<Float32Array>} Audio samples (float32, [-1, 1])
   */
  async decodeSentence(hiddenStates, T) {
    if (T === 0) return new Float32Array(0);

    const backbone = await this._runBackbone(hiddenStates, T);
    const audio = istft(backbone.data, backbone.T);

    // Trim to expected length: (T-1) * TOKEN_SIZE samples
    const expectedLen = (T - 1) * TOKEN_SIZE;
    if (audio.length >= expectedLen) {
      return audio.slice(audio.length - expectedLen);
    }
    return audio;
  }

  /**
   * Decode a full encoded speech to audio.
   * @param {object} encoded - Parsed .soprano file from loadSopranoFile()
   * @param {function} [onProgress] - Progress callback (sentenceIndex, totalSentences)
   * @returns {Promise<Float32Array>} Concatenated audio for all texts
   */
  async decode(encoded, onProgress = null) {
    const pieces = new Map(); // (textIndex, sentenceIndex) -> Float32Array

    for (let i = 0; i < encoded.sentences.length; i++) {
      const s = encoded.sentences[i];
      if (onProgress) onProgress(i, encoded.sentences.length);

      const audio = await this.decodeSentence(s.hiddenStates, s.T);
      pieces.set(`${s.textIndex}_${s.sentenceIndex}`, {
        audio,
        textIndex: s.textIndex,
        sentenceIndex: s.sentenceIndex,
      });
    }

    // Determine number of texts
    let numTexts = 0;
    for (const s of encoded.sentences) {
      numTexts = Math.max(numTexts, s.textIndex + 1);
    }

    // Concatenate sentences per text, then all texts
    const allAudio = [];
    for (let textIdx = 0; textIdx < numTexts; textIdx++) {
      const textPieces = [];
      for (const [, piece] of pieces) {
        if (piece.textIndex === textIdx) textPieces.push(piece);
      }
      textPieces.sort((a, b) => a.sentenceIndex - b.sentenceIndex);
      for (const piece of textPieces) {
        allAudio.push(piece.audio);
      }
    }

    // Merge into single Float32Array
    const totalLen = allAudio.reduce((sum, a) => sum + a.length, 0);
    const result = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of allAudio) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  }

  /**
   * Play decoded audio through the Web Audio API.
   * @param {Float32Array} audio - Audio samples
   * @param {number} [sampleRate=32000] - Sample rate
   * @returns {Promise<AudioBufferSourceNode>} The playing source node
   */
  play(audio, sampleRate = SAMPLE_RATE) {
    if (!this.audioContext) {
      this.audioContext = new AudioContext({ sampleRate });
    }
    const ctx = this.audioContext;
    const buffer = ctx.createBuffer(1, audio.length, sampleRate);
    buffer.getChannelData(0).set(audio);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();
    return source;
  }

  /**
   * Convert audio to a WAV blob.
   * @param {Float32Array} audio - Audio samples in [-1, 1]
   * @param {number} [sampleRate=32000]
   * @returns {Blob} WAV file blob
   */
  toWavBlob(audio, sampleRate = SAMPLE_RATE) {
    const numSamples = audio.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    // WAV header
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);         // PCM
    view.setUint16(20, 1, true);          // format = PCM
    view.setUint16(22, 1, true);          // mono
    view.setUint32(24, sampleRate, true);  // sample rate
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);          // block align
    view.setUint16(34, 16, true);         // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // PCM int16 data
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, audio[i]));
      view.setInt16(44 + i * 2, s * 32767, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }
}

export { SAMPLE_RATE, TOKEN_SIZE, HIDDEN_DIM };
