/** Use HTML5 constraints API to ask the user's browser if all fields are valid. */
export const checkAllFormControlValidity = (formId: string) => {
  const form = document.getElementById(formId) as HTMLFormElement;
  // if no form exists, then no input errors exist, so we say things are "valid"
  if (!form) return true;
  const allFormElements = Array.from(form.elements).filter((tag) =>
    [`select`, `textarea`, `input`].includes(tag.tagName.toLowerCase())
  ) as HTMLFormElement[];
  return allFormElements.every((f) => f.validity.valid);
};

/**
 * Use HTML5 constraints API to ask the user's browser if a specific field is valid.
 *
 * Returns `true` if the element does not exist in the document.
 */
export const checkFormControlValidity = (elementId: string) => {
  const formElement = document.getElementById(elementId) as HTMLFormElement;
  return formElement?.validity.valid ?? true;
};
