export const LIBRARY = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

// Byte-level BPE spells every byte as a printable character. This maps them back.
const BYTES = new Map();
for (let b = 0, extra = 0; b < 256; b++) {
  const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
  BYTES.set(String.fromCodePoint(printable ? b : 256 + extra++), b);
}
const utf8 = new TextDecoder();

/** How much of `buf` is whole UTF-8 characters, leaving out a character cut off at the end. */
function whole(buf) {
  for (let i = buf.length - 1; i >= Math.max(buf.length - 3, 0); i--) {
    // A continuation byte, so the character starts further back.
    if ((buf[i] & 0xc0) === 0x80) continue;
    const need = buf[i] >= 0xf0 ? 4 : buf[i] >= 0xe0 ? 3 : buf[i] >= 0xc0 ? 2 : 1;
    return buf[i] < 0xf8 && buf.length - i < need ? i : buf.length;
  }
  return buf.length;
}

/**
 * Each token's text from its bytes. A character split over tokens goes on the token that finishes it, and the bytes a
 * token holds of a character it doesn't finish show as escapes, so a split emoji isn't two replacement characters.
 */
export function spell(raw) {
  let carry = [];
  return raw.map((piece) => {
    const buf = Uint8Array.from([...carry, ...piece]);
    const cut = whole(buf);
    carry = [...buf.subarray(cut)];
    const part = piece.length ? carry.slice(-piece.length) : [];
    const escapes = part.map((b) => `\\x${b.toString(16).toUpperCase().padStart(2, "0")}`).join("");
    // A model can start a character and never finish it, and those stray bytes have nothing to draw.
    return utf8.decode(buf.subarray(0, cut)).replaceAll("\uFFFD", "") + escapes;
  });
}

/** `spell` for token ids from a byte-level BPE tokenizer, which Qwen3 and Granite both use, without special tokens. */
export function pieces(tok, ids) {
  // _tokenizer is private, though the library is pinned to one version above.
  const raw = ids.map((id) =>
    tok.all_special_ids.includes(id) ? [] : Array.from(tok._tokenizer.id_to_token(id), (c) => BYTES.get(c) ?? 63),
  );
  return spell(raw);
}

/** Splits a token into its text and the count of trailing bytes `spell` wrote as escapes. */
export function partial(text) {
  const at = text.search(/(?:\\x[0-9A-F]{2})+$/);
  return at < 0 ? [text, 0] : [text.slice(0, at), (text.length - at) / 4];
}

/**
 * Tokens as a reader should see them. A token holding only some bytes of a character shows that whole character, and
 * `note` explains, since escapes like \xF0\x9F\x9A read as a bug to anyone who hasn't met UTF-8.
 */
export function readable(tokens) {
  return tokens.map((text, i) => {
    const [head, bytes] = partial(text);
    if (!bytes) return { text, note: "" };
    const at = tokens.findIndex((t, k) => k > i && partial(t)[0]);
    if (at < 0)
      return { text: `${head}…`, note: `This token holds ${bytes} bytes of a character the text never finishes.` };
    const char = String.fromCodePoint(tokens[at].codePointAt(0));
    const total = new TextEncoder().encode(char).length;
    const by = at === i + 1 ? "The next token" : "A later token";
    return {
      text: head + char,
      note: `This token holds only ${bytes} of the ${total} bytes that spell ${char}, so it can't be drawn on its own. ${by} finishes it.`,
    };
  });
}
