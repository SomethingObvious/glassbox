"""Animated pictures of LLM inference, drawn from real runs of small public models."""

from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...


def _byte_level() -> dict[str, int]:
    """Byte-level BPE spells every byte as a printable character. This maps them back."""
    out, extra = {}, 0
    for b in range(256):
        if 33 <= b <= 126 or 161 <= b <= 172 or b >= 174:
            out[chr(b)] = b
        else:
            out[chr(256 + extra)] = b
            extra += 1
    return out


BYTES = _byte_level()


def _whole(buf: bytes) -> int:
    """How much of `buf` is whole UTF-8 characters, leaving out a character cut off at the end."""
    for i in range(len(buf) - 1, max(len(buf) - 4, -1), -1):
        if buf[i] & 0xC0 == 0x80:
            continue  # a continuation byte, so the character starts further back
        need = 4 if buf[i] >= 0xF0 else 3 if buf[i] >= 0xE0 else 2 if buf[i] >= 0xC0 else 1
        return i if buf[i] < 0xF8 and len(buf) - i < need else len(buf)
    return len(buf)


def spell(raw: Sequence[bytes]) -> list[str]:
    """Each token's text. A character split over tokens goes on the token that finishes it, and the
    bytes a token holds of a character it doesn't finish show as escapes, so an emoji split in two
    reads as `\\xF0\\x9F` then the emoji rather than two replacement characters."""
    out, carry = [], b""
    for piece in raw:
        buf = carry + piece
        cut = _whole(buf)
        carry = buf[cut:]
        part = carry[-len(piece) :] if piece else b""
        out.append(buf[:cut].decode(errors="replace") + "".join(f"\\x{b:02X}" for b in part))
    return out


def pieces(tok: Tokenizer, ids: list[int]) -> list[str]:
    """`spell` for token ids from a byte-level BPE tokenizer, which Granite and Qwen3 both use."""
    return spell([bytes(BYTES.get(c, 63) for c in t) for t in tok.convert_ids_to_tokens(ids)])
