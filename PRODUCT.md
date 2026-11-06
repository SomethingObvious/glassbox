# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Engineers and hiring peers who land here from the SomethingObvious GitHub
profile. They read code and know what a GPU is, though most have never watched
an inference engine work. They should leave after one screen believing the
author knows how LLM inference works, and they should click into a demo.

## Product Purpose

glassbox shows what happens inside LLM inference with real runs of small public
models. Each README banner is one animated SVG, and each one opens an
interactive version on this page. Success is a visitor who plays with a demo and
remembers it.

## Positioning

Nothing on the page is a mock-up. The routing picture is IBM Granite 3.0
1B-A400M's real router choices, the speculative decoding demo runs Qwen3-0.6B
and Qwen3-1.7B on the visitor's own GPU through WebGPU, and the waves demo
computes tiles and waves from public SM counts.

## Operating Context

Visitors arrive on desktop from a GitHub profile or README, often with a laptop
GPU. The draft demo downloads about 1.5 GB of models from Hugging Face on first
run, so it starts only when the visitor asks.

## Capabilities and Constraints

- The page has three demos, for speculative decoding (draft),
  mixture-of-experts routing (route) and GPU waves (waves).
- Plain ES modules with no build step, served by GitHub Pages from `docs/`.
  Transformers.js 4.3.1 comes from jsDelivr.
- A daily workflow routes the top Hacker News headline and publishes
  `route.svg` to the `live` branch.
- CI runs Ruff, pytest and Biome, and fails if a committed banner doesn't
  match its trace.

## Brand Commitments

The name is glassbox, in lower case. The author credit is Paul Wu, linking to
github.com/SomethingObvious.

## Evidence on Hand

- Traces in `docs/data` from real model runs.
- Banners in `assets/`: `route.svg`, `draft.svg` and `waves.svg`.
- No testimonials, metrics or press exist, and the page must not invent any.

## Product Principles

- Show the mechanism working, never a description of it.
- Every number on the page comes from a real run or a public spec.
- A demo works without an account, a server or a build step.
- Heavy work starts only when the visitor asks for it.
