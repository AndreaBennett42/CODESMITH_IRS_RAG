/**
 * Given a key (e.g. `/info/blah/blah`), returns the key prefixed with its
 * namespace (e.g. `info./info/blah/blah`).
 *
 * Does NOT check if the key actually exists. Use `i18n.exists(key)` for that.
 *
 * Historical note:
 * DF started with most keys in the flow under one giant (7,000 line) `info` namespace.
 * Later, with other namespaces, like `headings`, this function was added to help in
 * cases, like DFModal, where `info` had been hardcoded. (You couldn't add a modal to
 * a heading, because it would look in the wrong place for the modal text.)
 *
 * At a future point, en.yaml might get split into many files and that would be a great
 * opportunity to rethink the namespacing strategy in DF.
 */
export function getNamespacedKey(key: string): string {
  // if key includes a period, it's already namespaced
  if (key.includes(`.`)) {
    return key;
  }
  let nameSpace: string;
  if (key.startsWith(`/info`)) {
    nameSpace = `info.`;
  } else if (key.startsWith(`/heading`)) {
    nameSpace = `headings.`;
  } else if (key.startsWith(`/checkbox`)) {
    nameSpace = `fields.`;
  } else if (key.startsWith(`/iconList`)) {
    nameSpace = `iconLists.`;
  } else if (key.startsWith(`/modal`)) {
    nameSpace = `modals.`;
  } else {
    nameSpace = ``;
  }
  return `${nameSpace}${key}`;
}
