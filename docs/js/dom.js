export const $ = (id) => document.getElementById(id);
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
/** A colour token from style.css, so canvas drawings follow the light and dark sheets. */
export const tone = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
export const onScheme = (redraw) => matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);

const SVG = "http://www.w3.org/2000/svg";

/** Creates an SVG element with attributes and appends it to `parent`. */
export function svg(name, attrs, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent?.append(node);
  return node;
}

/** Sizes a canvas for the device pixel ratio and returns its context with the CSS width and height. */
export function canvas(node) {
  const width = node.getBoundingClientRect().width;
  // Setting node.height rewrites the attribute, so keep the CSS height from the first call.
  node.dataset.height ??= node.getAttribute("height");
  const height = Number(node.dataset.height);
  const dpr = devicePixelRatio || 1;
  node.width = width * dpr;
  node.height = height * dpr;
  node.style.height = `${height}px`;
  const ctx = node.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

export async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}
