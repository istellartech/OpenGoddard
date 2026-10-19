import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

// Pydocs generates the same docstrings at two bases. Keep only the current
// language's API tree, including pagination, after its middleware has run.
export const onRequest = defineRouteMiddleware(async (context, next) => {
  await next();
  const route = context.locals.starlightRoute;
  const english = route.lang === 'en';
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const otherBase = `${base}/${english ? '' : 'en/'}api/`;
  function filter(entries: typeof route.sidebar): typeof route.sidebar {
    return entries.flatMap((entry) => {
      if (entry.type === 'link') return entry.href.startsWith(otherBase) ? [] : [entry];
      const children = filter(entry.entries);
      return children.length ? [{ ...entry, entries: children }] : [];
    });
  }
  route.sidebar = filter(route.sidebar);
  for (const direction of ['prev', 'next'] as const) {
    if (route.pagination[direction]?.href.startsWith(otherBase)) route.pagination[direction] = undefined;
  }
});
