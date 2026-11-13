/**
 * The 4-bit copy of the LM head that the graphs score the vocabulary with, made here from the fp16 embedding table
 * so that the page does not have to download it. It comes out the same bytes as onnxruntime's MatMulNBits
 * quantizer makes for blocks of 32 without zero points, which `python -m switchboard head` checks its graphs with.
 */

const BLOCK = 32;

/** The value of an fp16 number from its bits. */
function half(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1024 + fraction) * 2 ** (exponent - 25);
}

/** The bits of the fp16 number nearest to minus an eighth of the fp16 number `bits`, with ties going to even. */
function scaleBits(bits) {
  const sign = (bits ^ 0x8000) & 0x8000;
  const exponent = (bits >> 10) & 31;
  if (exponent >= 4) return sign | ((bits & 0x7fff) - (3 << 10));
  // Too small to stay a normal number, so the fraction loses its low bits.
  const whole = exponent ? 1024 + (bits & 1023) : bits & 1023;
  const drop = exponent ? 4 - exponent : 3;
  const kept = whole >> drop;
  const rest = whole & ((1 << drop) - 1);
  const middle = 1 << (drop - 1);
  return sign | (kept + (rest > middle || (rest === middle && kept & 1) ? 1 : 0));
}

/**
 * The weights and then the scales of the 4-bit LM head for `table`, the fp16 embedding table as its bits with
 * `width` values a token. `pause` is awaited now and then so the page can draw.
 */
export async function scan(table, width, pause = async () => {}) {
  const values = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits++) values[bits] = half(bits);
  const vocab = table.length / width;
  const blocks = width / BLOCK;
  const out = new Uint8Array(vocab * (width / 2 + blocks * 2));
  const scales = new Uint16Array(out.buffer, vocab * (width / 2), vocab * blocks);
  let to = 0;
  for (let token = 0; token < vocab; token++) {
    if (token % 8192 === 0) await pause(token / vocab);
    for (let block = 0; block < blocks; block++) {
      const from = token * width + block * BLOCK;
      // The scale puts the value furthest from zero at -8, the end of the 4-bit range that reaches furthest.
      let low = from;
      let high = from;
      for (let i = from + 1; i < from + BLOCK; i++) {
        if (values[table[i]] < values[table[low]]) low = i;
        if (values[table[i]] > values[table[high]]) high = i;
      }
      const far = Math.abs(values[table[high]]) > Math.abs(values[table[low]]) ? table[high] : table[low];
      const scale = scaleBits(far);
      scales[token * blocks + block] = scale;
      // The quantizer divides by the scale as fp16 holds it, which is zero for a block of very small weights.
      const per = values[scale] ? Math.fround(1 / values[scale]) : 0;
      for (let i = from; i < from + BLOCK; i += 2) {
        const a = Math.fround(Math.fround(values[table[i]] * per) + 8);
        const b = Math.fround(Math.fround(values[table[i + 1]] * per) + 8);
        out[to++] =
          Math.max(0, Math.min(15, Math.floor(a + 0.5))) | (Math.max(0, Math.min(15, Math.floor(b + 0.5))) << 4);
      }
    }
  }
  return out;
}
