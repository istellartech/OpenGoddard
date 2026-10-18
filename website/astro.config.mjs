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
      description: '擬スペクトル法で最適制御問題を解くPythonライブラリ',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      head: [
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
        { tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+JP:wght@400;500;700;800&display=swap',
          },
        },
      ],
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
              sourceLink: { host: 'github', repo: 'istellartech/OpenGoddard', ref: 'master', root: '..' },
            },
          ],
          inventories: ['python'],
        }),
      ],
      sidebar: [
        { label: 'はじめに', link: '/' },
        {
          label: '基礎',
          items: ['theory/optimal-control', 'theory/methods', 'theory/pseudospectral'],
        },
        { label: '使い方', items: ['guide/usage', 'guide/troubleshooting'] },
        { label: '例題', items: [{ autogenerate: { directory: 'examples' } }] },
        { label: 'APIリファレンス', items: [pydocsSidebarGroup] },
      ],
    }),
  ],
});
