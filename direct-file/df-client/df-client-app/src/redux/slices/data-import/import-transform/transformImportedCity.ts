export function transformImportedCity(input: string | null | undefined): string {
  if (!input) {
    return ``;
  }
  const withoutPunctuation = input.replace(/[^a-zA-Z0-9\s/-]/g, ``);
  return withoutPunctuation.length >= 3 && withoutPunctuation.length <= 22 ? withoutPunctuation : ``;
}
