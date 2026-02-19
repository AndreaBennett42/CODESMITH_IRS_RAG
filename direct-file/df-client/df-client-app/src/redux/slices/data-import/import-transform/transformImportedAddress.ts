import { sanitizeString } from '../../../../components/FormControl/helpers.js';

export function transformImportedAddress(input: string | null | undefined): string {
  // Both the MeF schema and the Fact Graph disallow leading and trailing whitespaces
  // in street names, but our data import sources might send us addresses that have them.
  // This is where we correct that - before saving the /imported* facts to the Fact Graph.
  const address = input || ``;
  const withoutPunctionation = address.replace(/[^a-zA-Z0-9 /-]/g, ``);
  const sanitized = sanitizeString(withoutPunctionation);

  if (sanitized.isEmpty) {
    return ``;
  }

  return sanitized.value.substring(0, 35);
}
