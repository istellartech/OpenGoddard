import { defineEcConfig } from '@astrojs/starlight/expressive-code';

export default defineEcConfig({
  // Code components receive page URLs including the GitHub Pages base;
  // Markdown processing can instead supply a content file path.
  getBlockLocale: ({ file }) =>
    /(^|\/)en(\/|$)/.test(file.url?.pathname ?? file.path) ? 'en' : 'ja',
});
