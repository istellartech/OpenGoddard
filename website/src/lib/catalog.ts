// Read the learning order from examples/catalog.py so it is defined only once.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export interface ExampleEntry {
  id: string;
  directory: string;
  title: string;
  number: string;
}

export const repoRoot = resolve(process.cwd(), '..');
export const githubTree = 'https://github.com/istellartech/OpenGoddard/tree/master';

export function examples(): ExampleEntry[] {
  const source = readFileSync(resolve(repoRoot, 'examples/catalog.py'), 'utf-8');
  const pattern = /Example\(\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",?\s*\)/g;
  return [...source.matchAll(pattern)].map(([, id, directory, title]) => ({
    id,
    directory,
    title,
    number: directory.slice(0, 2),
  }));
}

export function findExample(id: string): ExampleEntry {
  const entry = examples().find((example) => example.id === id);
  if (!entry) throw new Error(`unknown example id: ${id}`);
  return entry;
}

export interface MediaManifest {
  id: string;
  videos: { name: string; video: string; poster: string }[];
  figures: string[];
  comparison: string | null;
  data: string[];
  solver?: Record<string, { success: boolean; message: string }>;
}

export function mediaManifest(id: string): MediaManifest | undefined {
  const path = resolve(process.cwd(), 'public/media', id, 'manifest.json');
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf-8')) : undefined;
}

export function mediaUrl(...parts: string[]): string {
  return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/media/${parts.join('/')}`;
}
