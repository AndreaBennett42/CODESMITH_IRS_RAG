import { ConcretePath } from '@irs/js-factgraph-scala';
import { Path } from '../../fact-dictionary/Path.js';

export function buildFormControlId<Subfield extends string>(concretePath: ConcretePath, subfield?: Subfield) {
  // This feels silly, but provides some security if we ever decide on a different naming scheme, for some reason
  const optionalSubfield = subfield ? `__` + subfield : ``;
  return `id-${concretePath}${optionalSubfield}` as const;
}

export function buildControlErrorId<Subfield extends string>(
  controlId: ReturnType<typeof buildFormControlId>,
  subfield?: Subfield
) {
  const optionalSubfield = subfield ? `__` + subfield : ``;
  return `${controlId}${optionalSubfield}__error-msg` as const;
}

export function buildHintId<Subfield extends string>(
  controlId: ReturnType<typeof buildFormControlId> | string,
  subfield?: Subfield
) {
  const optionalSubfield = subfield ? `__` + subfield : ``;
  return `${controlId}${optionalSubfield}__hint` as const;
}

export function buildHintKey<Subfield extends string>(path?: Path, subField?: Subfield) {
  if (path) {
    const optionalSubfield = subField ? `.${subField}` : ``;
    return `info./info${path}${optionalSubfield}.helpText.hint`; //Translation.tsx will add the .text to the final key
  } else {
    return ``;
  }
}

// Edge case: Email field is readonly and uses a DFmodal as the hint.
export function buildReadonlyHintKey(path?: Path) {
  if (path) {
    return `info.${path}.readOnlyField`; // DFModal component will add the .helpText.modals to the final key
  } else {
    return ``;
  }
}

/* Return the original string, and set empty boolean
 */
export function sanitizeString(inputString: string) {
  // Trims trailing and leading whitespaces, collapses adjacent spaces
  const value = inputString.trim().replace(/ +/g, ` `);
  return { value, isEmpty: value === `` };
}
