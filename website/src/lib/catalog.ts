// Read the learning order from examples/catalog.py so it is defined only once.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { localePrefix } from './i18n';

export interface ExampleEntry {
  id: string;
  stem: string;
  title: string;
  number: string;
}

export const repoRoot = resolve(process.cwd(), '..');
export const githubTree = 'https://github.com/istellartech/OpenGoddard/tree/master';

export function examples(): ExampleEntry[] {
  const source = readFileSync(resolve(repoRoot, 'examples/catalog.py'), 'utf-8');
  const pattern = /Example\(\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",?\s*\)/g;
  return [...source.matchAll(pattern)].map(([, id, stem, title]) => ({
    id,
    stem,
    title,
    number: stem.slice(0, 2),
  }));
}

export function findExample(id: string): ExampleEntry {
  const entry = examples().find((example) => example.id === id);
  if (!entry) throw new Error(`unknown example id: ${id}`);
  return entry;
}

/** Numbers written by scripts/build_media.py for one example. */
export type Summary = Record<string, any> & {
  success: boolean;
  message: string;
  objective: number;
};

export function summary(id: string): Summary | undefined {
  const path = resolve(process.cwd(), 'public/media', id, 'summary.json');
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf-8')) : undefined;
}

export function hasMedia(...parts: string[]): boolean {
  return existsSync(resolve(process.cwd(), 'public/media', ...parts));
}

export function mediaUrl(...parts: string[]): string {
  return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/media/${parts.join('/')}`;
}

export function exampleHref(id: string, locale?: string): string {
  const example = findExample(id);
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}/${localePrefix(locale)}examples/${example.number}-${example.id.replaceAll('_', '-')}/`;
}
