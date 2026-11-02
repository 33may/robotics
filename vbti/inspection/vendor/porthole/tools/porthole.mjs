#!/usr/bin/env node
/**
 * porthole — install and update registry components you own.
 *
 * The whole design rests on one decision: **we keep the base.**
 *
 * When a component is installed, a pristine copy of exactly what was installed
 * is recorded under `.porthole/base/`. That copy is what makes `update` a real
 * three-way merge (`git merge-file yours base upstream`) instead of a guess.
 * Only hunks that overlap your edits can conflict; upstream fixes to untouched
 * regions land silently.
 *
 * This is the difference between owning source and being stranded on it. Tools
 * that record no base — shadcn's `diff` among them — can only show a two-way
 * diff in which every intentional local edit is indistinguishable from an
 * upstream change, which is why nobody uses them twice.
 *
 *   porthole add <item> [--into DIR]
 *   porthole status [--into DIR]
 *   porthole update <item|--all> [--into DIR]
 *   porthole list
 */

import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));
const REGISTRY_ROOT = join(REPO_ROOT, 'packages/registry');
const REGISTRY_SRC = join(REGISTRY_ROOT, 'src');

/**
 * Chromium's taxonomy, adopted verbatim. Recording *how* a component is meant
 * to be updated turns "did someone fork this?" from tribal knowledge into a
 * queryable fact, and forces a named owner onto every hard fork.
 */
const UPDATE_MECHANISMS = ['Auto', 'Manual', 'Static', 'Static.HardFork'];

// ── helpers ─────────────────────────────────────────────────────────────────

function readRegistry() {
  return JSON.parse(readFileSync(join(REGISTRY_ROOT, 'registry.json'), 'utf8'));
}

function registryRevision() {
  try {
    return execFileSync('git', ['rev-parse', '--short', 'HEAD'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    }).trim();
  } catch {
    return 'unknown';
  }
}

function manifestPath(target) {
  return join(target, '.porthole/manifest.json');
}

function readManifest(target) {
  const path = manifestPath(target);
  if (!existsSync(path)) return { registry: 'porthole', items: {} };
  return JSON.parse(readFileSync(path, 'utf8'));
}

function writeManifest(target, manifest) {
  const path = manifestPath(target);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
}

function basePath(target, item, file) {
  return join(target, '.porthole/base', item, file);
}

function copyInto(from, to) {
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
}

function sameBytes(a, b) {
  if (!existsSync(a) || !existsSync(b)) return false;
  return readFileSync(a).equals(readFileSync(b));
}

function fail(message) {
  console.error(`porthole: ${message}`);
  process.exit(1);
}

// ── commands ────────────────────────────────────────────────────────────────

function commandList() {
  const registry = readRegistry();
  for (const [name, item] of Object.entries(registry.items)) {
    console.log(`${name.padEnd(14)} ${item.description}`);
  }
}

function commandAdd(itemName, target) {
  const registry = readRegistry();
  const item = registry.items[itemName];
  if (!item) fail(`unknown item "${itemName}" — try: porthole list`);

  const manifest = readManifest(target);
  if (manifest.items[itemName]) {
    fail(`"${itemName}" is already installed — use: porthole update ${itemName}`);
  }

  for (const file of item.files) {
    const source = join(REGISTRY_SRC, file);
    if (!existsSync(source)) fail(`registry file missing: ${file}`);
    copyInto(source, join(target, file));
    // The base copy. Everything downstream depends on this existing.
    copyInto(source, basePath(target, itemName, file));
  }

  manifest.items[itemName] = {
    version: registryRevision(),
    files: item.files,
    updateMechanism: 'Auto',
    localModifications: 'None',
  };
  writeManifest(target, manifest);

  console.log(`added ${itemName} (${item.files.length} file(s)) at ${registryRevision()}`);
  if (item.npmDependencies?.length) {
    console.log(`  needs: npm i ${item.npmDependencies.join(' ')}`);
  }
}

function commandStatus(target) {
  const manifest = readManifest(target);
  const entries = Object.entries(manifest.items);
  if (entries.length === 0) {
    console.log('no components installed');
    return;
  }

  const revision = registryRevision();
  for (const [name, entry] of entries) {
    const edited = entry.files.filter(
      (file) => !sameBytes(join(target, file), basePath(target, name, file)),
    );
    const upstreamMoved = entry.files.filter(
      (file) => !sameBytes(join(REGISTRY_SRC, file), basePath(target, name, file)),
    );

    const flags = [
      edited.length ? `forked(${edited.length})` : 'pristine',
      upstreamMoved.length ? 'upstream-changed' : 'current',
      entry.updateMechanism,
    ].join(' ');
    console.log(`${name.padEnd(14)} ${entry.version.padEnd(10)} ${flags}`);
  }
  console.log(`\nregistry at ${revision}`);
}

function commandUpdate(itemName, target) {
  const manifest = readManifest(target);
  const entry = manifest.items[itemName];
  if (!entry) fail(`"${itemName}" is not installed`);

  if (entry.updateMechanism === 'Static' || entry.updateMechanism === 'Static.HardFork') {
    console.log(`${itemName}: updateMechanism is ${entry.updateMechanism} — skipping`);
    return;
  }

  let conflicted = 0;
  let merged = 0;

  for (const file of entry.files) {
    const yours = join(target, file);
    const base = basePath(target, itemName, file);
    const upstream = join(REGISTRY_SRC, file);

    if (!existsSync(base)) fail(`missing base for ${file} — cannot merge without it`);
    if (!existsSync(upstream)) {
      console.log(`  ${file}: removed upstream — left in place, decide manually`);
      continue;
    }
    if (sameBytes(base, upstream)) continue; // upstream unchanged

    // git merge-file writes the merge into `yours` in place and exits with the
    // number of conflicts — so a conflicted merge is a nonzero exit, which
    // execFileSync reports by throwing. That is a result, not a failure.
    let conflicts = 0;
    try {
      execFileSync(
        'git',
        ['merge-file', '-L', 'yours', '-L', 'base', '-L', 'upstream', yours, base, upstream],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] },
      );
    } catch (error) {
      if (typeof error.status !== 'number' || error.status < 0) {
        fail(`merge failed for ${file}: ${error.message}`);
      }
      conflicts = error.status;
    }

    if (conflicts > 0) {
      conflicted += conflicts;
      console.log(`  ${file}: ${conflicts} conflict(s) — markers left in the file`);
      // Deliberately do NOT advance the base: the file still contains conflict
      // markers, so recording it as pristine upstream would destroy the only
      // reference point a re-run after resolution has.
      continue;
    }

    merged += 1;
    console.log(`  ${file}: merged`);
    copyInto(upstream, base);
  }

  entry.version = registryRevision();
  writeManifest(target, manifest);
  console.log(
    `${itemName}: ${merged} file(s) merged, ${conflicted} conflict(s), now at ${entry.version}`,
  );
}

function commandUpdateAll(target) {
  const manifest = readManifest(target);
  for (const name of Object.keys(manifest.items)) commandUpdate(name, target);
}

// ── entry ───────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const intoIndex = argv.indexOf('--into');
const target = resolve(intoIndex === -1 ? process.cwd() : argv[intoIndex + 1]);
const positional = argv.filter((arg, index) => {
  if (arg === '--into') return false;
  if (intoIndex !== -1 && index === intoIndex + 1) return false;
  return !arg.startsWith('--');
});

const [command, argument] = positional;

switch (command) {
  case 'list':
    commandList();
    break;
  case 'add':
    if (!argument) fail('usage: porthole add <item> [--into DIR]');
    commandAdd(argument, target);
    break;
  case 'status':
    commandStatus(target);
    break;
  case 'update':
    if (argv.includes('--all')) commandUpdateAll(target);
    else if (argument) commandUpdate(argument, target);
    else fail('usage: porthole update <item|--all> [--into DIR]');
    break;
  default:
    console.log(
      [
        'porthole — install and update registry components you own',
        '',
        '  porthole list',
        '  porthole add <item>      [--into DIR]',
        '  porthole status          [--into DIR]',
        '  porthole update <item>   [--into DIR]',
        '  porthole update --all    [--into DIR]',
        '',
        `updateMechanism values: ${UPDATE_MECHANISMS.join(', ')}`,
      ].join('\n'),
    );
}
