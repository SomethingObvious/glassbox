import { $, calm } from "./dom.js";
import { initDraft } from "./draft.js";
import { initLens } from "./lens.js";
import { initQuant } from "./quant.js";
import { initRoute } from "./route.js";

// A figure's rules are ruled across the sheet the first time it comes into view. Without this script they are
// simply there. Figure 1 is in view from the start, so the stylesheet draws its rules as the page loads.
if (!calm) {
  const ruler = new IntersectionObserver(
    (entries) => {
      for (const { target, isIntersecting } of entries) {
        if (!isIntersecting) continue;
        target.classList.remove("unruled");
        ruler.unobserve(target);
      }
    },
    { threshold: 0.12 },
  );
  for (const figure of document.querySelectorAll(".part .fig")) {
    figure.classList.add("unruled");
    ruler.observe(figure);
  }
}

// The running head marks the part that is on screen.
const links = new Map(
  [...document.querySelectorAll(".running nav a")].map((link) => [link.getAttribute("href").slice(1), link]),
);
const reader = new IntersectionObserver(
  (entries) => {
    for (const { target, isIntersecting } of entries) {
      if (!isIntersecting) continue;
      for (const [id, link] of links) {
        if (id === target.id) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      }
    }
  },
  // A part is the current one while it crosses a line a third of the way down the window.
  { rootMargin: "-33% 0px -66% 0px" },
);
for (const id of links.keys()) reader.observe($(id));

// What comes after the opening stays unseen until every figure is laid out, which the stylesheet sees to.
await Promise.all([initDraft(), initRoute(), initQuant(), initLens()]).finally(() =>
  $("draft").classList.remove("unsized"),
);
window.__done = true;
