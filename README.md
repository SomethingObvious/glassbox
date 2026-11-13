# switchboard

[![Check](https://github.com/SomethingObvious/switchboard/actions/workflows/check.yml/badge.svg)](https://github.com/SomethingObvious/switchboard/actions/workflows/check.yml)

Animated pictures of what happens inside LLM inference, drawn from real runs of small public models. Every banner is a single SVG with CSS animation, so it plays in a README with no JavaScript, and each one opens an [interactive version](https://somethingobvious.github.io/switchboard/) where the speculative decoding demo runs two real models on your own GPU through WebGPU.

<a href="https://somethingobvious.github.io/switchboard/#route"><img src="https://raw.githubusercontent.com/SomethingObvious/switchboard/live/route.svg" width="100%" alt="Today's top Hacker News headline going through a mixture-of-experts router, each token picking 8 of 32 experts in every layer"></a>

IBM Granite 3.0 1B-A400M's router sends every token to 8 of its 32 experts in each of 24 layers. Every morning a [workflow](.github/workflows/headline.yml) routes the top Hacker News headline through one real forward pass, publishes the banner and its trace to the `live` branch, and the page opens Figure 2 on it. You can also type your own sentence there. It runs an 8-bit copy of the model on your GPU, which picks the same experts as the full one at 94.8% of token and layer pairs.

<a href="https://somethingobvious.github.io/switchboard/#draft"><img src="https://raw.githubusercontent.com/SomethingObvious/switchboard/live/draft.svg" width="100%" alt="Qwen3-1.7B writing with an EAGLE-3 head guessing 4 tokens at a time, and the same answer written by Qwen3-1.7B alone underneath on the same clock"></a>

The same workflow then has Qwen3-1.7B ask three questions about that headline, with an EAGLE-3 head I trained for it. (It asks because a model this small makes the story up when it is told to explain one, and if its questions name anything the headline doesn't, that day's run is the saved prompt instead.) The head is one decoder layer that reads the big model's hidden states and guesses 4 tokens, and Qwen3-1.7B checks all 4 in one pass, so the text is exactly what it writes alone. Both ways are timed pass by pass on the 2 CPU cores of GitHub's free runner, where the first run wrote the same 42 tokens 1.38 times as fast. On 4 cores of my laptop it runs 1.6 to 1.9 times as fast.

Getting a speedup on a CPU at all took packing every weight for oneDNN once, the way `torch.compile` does when it freezes a model. Plain `F.linear` repacks the weight on every call with more than one row, so checking 5 tokens cost twice as much as writing 1 and the guesses never paid for themselves. Packed, both cost about 170 ms on 4 threads.

The head started from Tencent's [AngelSlim head](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) and trained for 90 minutes on an RTX 4070 laptop GPU, on Qwen3-1.7B's own answers to 6,837 Hacker News titles and 2,500 [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) prompts. On held-out prompts its first guess went from 44% to 64% right. The weights are at [hoyporti/qwen3-1.7b-eagle3-switchboard](https://huggingface.co/hoyporti/qwen3-1.7b-eagle3-switchboard) and the scripts are in [training](training).

The page's live button runs the same race in your browser over WebGPU. For that, `python -m switchboard head` builds two graphs from the 4-bit browser build of Qwen3-1.7B, one that writes alone and one that runs the head (in 4 bits) right after the big model in the same call, both made for WebGPU graph capture. A fourth row triples the pass on WebGPU, so the live run guesses two tokens a pass, and the head makes the second from its own output for the first. On an RTX 4070 laptop GPU it comes out about 1.45 times as fast as writing alone on the default prompt, at about 230 tokens a second against 160, and 1.4 times as fast on average over one day's top 40 Hacker News headlines (223 against 160). How much it helps depends on how easy the answer is to guess, and the prompts under the figure run from about 0.9 on a translation into French to 2.2 on a repeated sentence, which is written at about 350 tokens a second.

The WebGPU ArgMax in onnxruntime-web 1.31.1 walks the whole vocabulary in one shader thread, which was 8.7 ms of a 20 ms pass, so the graphs pick the token with a max in two steps instead, which brought a pass alone to 11.5 ms. Its fused 4-bit kernels also only run for one row, so the page loads its own build of the wasm from [docs/ort](docs/ort), which keeps them for up to four. Its attention also adds up in a different order when a pass has two or three rows, which is enough to tip a near tie between two tokens, and the same build makes it add up the way it does for one. Even then a pass with three rows took 14.6 ms against 9.2 for one, as the 4-bit kernels read their input from workgroup memory again for every few output columns. The build has each thread keep its part of the input in registers (on NVIDIA GPUs only, as it made an AMD integrated GPU slower), which brought the three-row pass to 11.2 ms. A pick takes about 2 ms to get back to the CPU and the GPU would sit idle for it, so the pass graph works out which rows the next pass reads (it needs to know whether the guess was kept) and the page keeps two passes on the GPU at once, which makes a pass alone about 9.2 ms.

By then the largest thing left in a pass was the LM head, which is the fp16 embedding table on its side, 622 MB that every pass read in full. The graphs now score the vocabulary with a 4-bit copy of it and score only the 1,024 likeliest tokens again from the fp16 table. The page makes that copy itself from the table it has just downloaded, which takes a couple of seconds and comes out the same bytes as onnxruntime's quantizer writes, so it adds nothing to the 1.5 GB download. That wrote the same 1,627 tokens over the 40 headlines, and it took a pass alone to 6.6 ms and one with guesses from 11.2 to 8.5. The norms on attention's q and k are only 128 values a head, and the stock layer norm gave each of those to one shader thread while the rest of its workgroup waited. In the build every thread adds one up itself, which took another 0.3 ms off both passes. If the guessed run has taken longer after 24 passes than the same tokens took alone, the rest of the answer is written without the head.

The page also has two figures with no banners. Figure 3 rounds a real row of Qwen3-0.6B's weights to int8, fp8, int4, int3 or int2 in the browser, next to what the whole model writes in each format. In int4 with groups of 128 it splits from bf16 after two tokens, and in int3 it stops making sense. Figure 4 reads Qwen3-0.6B's next-token guess out of every layer. Asked for the French word for dog, its last few layers guess 犬 before settling on chien.

## Running It

The traces in `docs/data` are committed, so redrawing a banner needs only Python 3.11 or newer. Tracing again downloads the models and runs them on MPS or CPU.

```sh
uv run python -m switchboard route            # redraw from the saved trace
uv run --extra trace python -m switchboard draft --trace
uv run --extra trace python -m switchboard route --headline live/route.svg
uv run --extra trace python -m switchboard quant          # Figure 3, every weight format on CPU
uv run --extra trace python -m switchboard lens           # Figure 4, the logit lens on CPU
uv run --extra export python -m switchboard routes       # the 8-bit router model Figure 2 runs in the browser
uv run --extra export python -m switchboard head         # the two graphs Figure 1 runs in the browser
```

The page in `docs/` is plain ES modules with no build step, served by GitHub Pages. CI runs Ruff, pytest and Biome, and fails if a committed banner doesn't match its trace.
