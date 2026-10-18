// @ts-check
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';

export default defineConfig({
  site: 'https://istellartech.github.io',
  base: '/OpenGoddard',
  markdown: {
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
  },
  integrations: [
    starlight({
      title: 'OpenGoddard',
      description: '擬スペクトル法による最適軌道計算のPythonライブラリ',
      defaultLocale: 'root',
      locales: { root: { label: '日本語', lang: 'ja' } },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/istellartech/OpenGoddard' },
      ],
      customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
      plugins: [
        starlightPydocs({
          packages: [
            {
              name: 'OpenGoddard',
              search: ['../src'],
              docstringStyle: 'google',
            },
          ],
          inventories: ['python'],
        }),
      ],
      sidebar: [
        { label: 'はじめに', items: ['index', 'getting-started', 'concepts/pseudospectral'] },
        { label: '例題', items: [{ autogenerate: { directory: 'examples' } }] },
        { label: 'APIリファレンス', items: [pydocsSidebarGroup] },
      ],
    }),
  ],
});
