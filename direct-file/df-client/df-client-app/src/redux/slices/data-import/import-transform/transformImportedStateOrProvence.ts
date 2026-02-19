export function transformImportedStateOrProvence(input: string | null | undefined): string {
  if (!input) {
    return ``;
  }
  const withoutPunctuation = input.replace(/[^a-zA-Z0-9\s/-]/g, ``);
  return withoutPunctuation.length === 2 ? withoutPunctuation : ``;
}
