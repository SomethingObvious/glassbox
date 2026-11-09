# glassbox

[![Check](https://github.com/SomethingObvious/glassbox/actions/workflows/check.yml/badge.svg)](https://github.com/SomethingObvious/glassbox/actions/workflows/check.yml)

Animated pictures of what happens inside LLM inference, drawn from real runs of small public models. Every banner is a single SVG with CSS animation, so it plays in a README with no JavaScript, and each one opens an [interactive version](https://somethingobvious.github.io/glassbox/) where the speculative decoding demo runs two real models on your own GPU through WebGPU.

<a href="https://somethingobvious.github.io/glassbox/#route"><img src="https://raw.githubusercontent.com/SomethingObvious/glassbox/live/route.svg" width="100%" alt="Today's top Hacker News headline going through a mixture-of-experts router, each token picking 8 of 32 experts in every layer"></a>

IBM Granite 3.0 1B-A400M's router sends every token to 8 of its 32 experts in each of 24 layers. Every morning a [workflow](.github/workflows/headline.yml) routes the top Hacker News headline through one real forward pass, publishes the banner and its trace to the `live` branch, and the page opens Figure 2 on it.

<a href="https://somethingobvious.github.io/glassbox/#draft"><img src="assets/draft.svg" width="100%" alt="Qwen3-0.6B guessing tokens and Qwen3-1.7B checking four at a time"></a>

Qwen3-0.6B guesses 4 tokens and Qwen3-1.7B checks them in one forward pass. Decoding is greedy, so the text is exactly what Qwen3-1.7B writes alone, here in 16 big-model passes instead of 48.

<a href="https://somethingobvious.github.io/glassbox/#waves"><img src="assets/waves.svg" width="100%" alt="A matrix multiply's 448 tiles going out to an H100's 132 SMs in 4 waves"></a>

A GEMM's 448 output tiles go out to an H100's 132 SMs in waves, and the last wave leaves 80 of them idle.

## Running It

The traces in `docs/data` are committed, so redrawing a banner needs only Python 3.11 or newer. Tracing again downloads the models and runs them on MPS or CPU.

```sh
uv run python -m glassbox route            # redraw from the saved trace
uv run --extra trace python -m glassbox draft --trace
uv run --extra trace python -m glassbox route --headline live/route.svg
```

The page in `docs/` is plain ES modules with no build step, served by GitHub Pages. CI runs Ruff, pytest and Biome, and fails if a committed banner doesn't match its trace.
