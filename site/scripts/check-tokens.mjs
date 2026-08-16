#!/usr/bin/env node
/**
 * Assert that lib/tokens.ts and app/globals.css carry the same colour values.
 *
 * These two files must agree because they feed two different renderers: the
 * website reads the CSS custom properties, while the Remotion composition reads
 * the TypeScript constants (Remotion's bundler cannot see globals.css). If they
 * drift, the explainer video quietly stops matching the product it explains.
 *
 * Run: npm run check:tokens
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const TS = resolve(HERE, "../lib/tokens.ts");
const CSS = resolve(HERE, "../app/globals.css");

/** Pull the TOKEN_HEXES map out of tokens.ts without importing TypeScript. */
function readTsTokens() {
  const src = readFileSync(TS, "utf8");
  const block = src.slice(src.indexOf("export const TOKEN_HEXES"));
  const out = {};
  const skipped = [];

  // Every key in the map, whether or not we can resolve its value. Compared
  // against `out` below so a parse miss is loud instead of silent — an earlier
  // version of this regex quietly dropped surface.bg2/.s1/.s2/.s3 because it
  // disallowed digits in the member name, and reported "OK" on 16 of 20.
  const declared = [...block.matchAll(/^\s*"?([a-zA-Z0-9-]+)"?:/gm)].map((m) => m[1]);

  // matches:  key: someObj.member,   and   "key-with-dash": someObj.member2,
  for (const [, rawKey, ref] of block.matchAll(
    /^\s*"?([a-zA-Z0-9-]+)"?:\s*([a-zA-Z][a-zA-Z0-9]*\.[a-zA-Z][a-zA-Z0-9]*),/gm,
  )) {
    const [obj, member] = ref.split(".");
    // resolve the reference back to its literal in the same file
    const objBody = src.match(
      new RegExp(`export const ${obj} = \\{([\\s\\S]*?)\\} as const;`),
    );
    if (!objBody) {
      skipped.push(`${rawKey} (no object literal named ${obj})`);
      continue;
    }
    const lit = objBody[1].match(new RegExp(`\\b${member}:\\s*"([^"]+)"`));
    if (!lit) {
      skipped.push(`${rawKey} (${obj}.${member} is not a string literal)`);
      continue;
    }
    out[rawKey] = lit[1].toLowerCase();
  }

  const unresolved = declared.filter((k) => !(k in out));
  if (unresolved.length || skipped.length) {
    console.error("check-tokens could not resolve every entry in TOKEN_HEXES:");
    for (const k of new Set([...unresolved, ...skipped])) console.error("  · " + k);
    console.error("\nThe check is only meaningful if it reads all of them. Fix the parser.");
    process.exit(2);
  }

  return out;
}

/** Pull --color-<name>: <hex> pairs out of the @theme block in globals.css. */
function readCssTokens() {
  const src = readFileSync(CSS, "utf8");
  const out = {};
  for (const [, name, value] of src.matchAll(
    /^\s*--color-([a-zA-Z0-9-]+):\s*(#[0-9a-fA-F]{3,8})\s*;/gm,
  )) {
    out[name] = value.toLowerCase();
  }
  return out;
}

const ts = readTsTokens();
const css = readCssTokens();

const problems = [];
const tsKeys = Object.keys(ts);

if (tsKeys.length === 0) {
  problems.push("could not parse any tokens out of lib/tokens.ts — check the regex");
}

for (const key of tsKeys) {
  if (!(key in css)) {
    problems.push(`--color-${key} is in tokens.ts but missing from globals.css`);
  } else if (css[key] !== ts[key]) {
    problems.push(`--color-${key}: globals.css has ${css[key]}, tokens.ts has ${ts[key]}`);
  }
}

if (problems.length) {
  console.error("TOKEN DRIFT — the site and the explainer video would disagree:\n");
  for (const p of problems) console.error("  · " + p);
  console.error("");
  process.exit(1);
}

console.log(`tokens OK — ${tsKeys.length} colours agree between tokens.ts and globals.css`);
