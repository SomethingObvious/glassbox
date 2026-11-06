---
name: glassbox
description: Live figures of LLM inference, set as a research paper with patent-drawing plates.
colors:
  sheet: "#fdfdfb"
  ink: "#121212"
  soft: "#555552"
  faint: "#d6d6d1"
  wash: "#f1f1ec"
  lost: "#c4161c"
  sheet-dark: "#131312"
  ink-dark: "#ecebe4"
  soft-dark: "#a9a8a0"
  faint-dark: "#3a3a37"
  wash-dark: "#1d1d1b"
  lost-dark: "#ff6b6b"
typography:
  display:
    fontFamily: "Libre Caslon Text, Iowan Old Style, Georgia, serif"
    fontSize: "clamp(32px, 3.4vw, 44px)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.015em"
  section:
    fontFamily: "Libre Caslon Text, Iowan Old Style, Georgia, serif"
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Libre Caslon Text, Iowan Old Style, Georgia, serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.6
  body-phone:
    fontFamily: "Libre Caslon Text, Iowan Old Style, Georgia, serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.6
  field:
    fontFamily: "Public Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
  figure:
    fontFamily: "Public Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Public Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    letterSpacing: "0.24em"
  readout:
    fontFamily: "Public Sans, Helvetica Neue, Arial, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    lineHeight: 1.2
rounded:
  none: "0"
spacing:
  gap: "8px"
  callout: "22px"
  column: "24px"
components:
  button:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
    height: "44px"
  button-go:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.sheet}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
    height: "44px"
  token:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "4px 9px"
    height: "36px"
---

# Design System: glassbox

## Overview

The page reads as a short research paper. The text column carries the argument
in a book serif, and each figure is a plate drawn in black ink on a white sheet,
with numbered callouts the way a technical drawing labels its parts. Nothing is
decorative. Every mark in a figure stands for something from a real run or a
public spec, and the reader can drive the figures.

## Colors

There is one ink and one sheet. Greys only separate layers of the same ink.

### Primary

- `ink` draws text, rules, chosen experts, busy SMs and the primary button.
- `sheet` is the page and the text on the primary button.

### Tertiary

- `lost` marks work that was thrown away or wasted, and nothing else. It shows
  up on the first rejected draft token, on idle SMs in the last wave and as the
  caret colour.

### Neutral

- `soft` is for secondary text (captions, side notes and axis labels), at 7.3:1
  on the sheet.
- `faint` is for cell outlines, the progress track and token chip borders.
- `wash` is the hover fill.

### Named Rules

- Red means lost work. A new state that isn't waste gets ink, a dash pattern or
  a weight, never a second hue.
- Dark mode uses its own steps from the same set (the `-dark` tokens), chosen
  for contrast on the dark sheet rather than inverted.
- Canvas figures read the tokens at draw time through `tone()` and redraw when
  the colour scheme changes.

## Typography

Libre Caslon Text sets the paper itself, from the title down to the body. Public
Sans letters everything inside a figure, which covers the captions, callouts,
readouts, controls and axis labels. Public Sans has plain zeros, which matters
in a page full of numbers.

### Hierarchy

- Title and section heads use `display` and `section`, balanced.
- Body copy uses `body` at a 62ch measure, and `body-phone` below 600px.
- The prompt box uses `field`, since iPhone Safari zooms into any field
  under 16px.
- Figure text uses `figure`. Readout numbers use `readout` with tabular figures.
- The abstract label and similar small heads use `label`, uppercase and tracked.

## Layout

The opening is a two-column spread, with the paper's front matter on the left
and Figure 1 on the right. Each later part is a 4fr/9fr grid, text then figure.
Below 960px the parts stack, and below 600px the readouts fold to two columns
and the running head drops its date. Figure 2's SVG resizes its labels so they
stay 12px on screen at any width.

## Elevation & Depth

The page is flat. Rules (2px around a figure's plate and 1px around a readout)
do the work that shadows would do elsewhere. The one exception is the primary
button's hover, a short red underline drawn with a box shadow.

## Shapes

Every corner is square. Draft tokens are boxes whose border style says what
happened to them: dashed for a guess, solid for kept, red dashed and struck for
the first miss, and an underline for the big model's own token. SMs are filled
squares when busy and red dashed outlines when idle.

## Components

- Buttons are 44px tall with a 2px ink border. The primary action (`button-go`)
  is filled ink. A disabled button switches to a dashed border.
- Figure callouts are a list of swatch, bold reference numeral and text.
  Numerals run 102 to 108 for Figure 1, 202 to 206 for Figure 2 and 302 to 304
  for Figure 3.
- A readout is a ruled grid of label and number pairs. When a progress bar sits
  right above it, the bar's track replaces its top rule.
- Token chips in Figure 2 are 36px tall on a mouse and at least 44px square on a
  coarse pointer. The current token gets a 2px ink border and bold text.

## Do's and Don'ts

- Do label every figure part with a callout numeral and explain it in the
  callout list.
- Do keep numbers tabular in readouts and on axes.
- Don't round corners, add gradients or add a second accent colour.
- Don't put patent claims, filing numbers or legal language on the page. The
  plates borrow the drawing style only.
