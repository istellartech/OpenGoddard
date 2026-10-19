/** The root locale is Japanese; English pages live below /en/. */
export function isEnglish(locale: string | undefined): boolean {
  return locale === 'en';
}

export function localePrefix(locale: string | undefined): string {
  return isEnglish(locale) ? 'en/' : '';
}
