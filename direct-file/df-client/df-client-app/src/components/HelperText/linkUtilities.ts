const isValidURL = (string: string) => {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
};

const isExternalLink = (url: string) => {
  try {
    const link = new URL(url);
    // TODO: we should check, once we're deployed in multiple environments that
    // this properly flags our meals as external.
    // We want to mark anything that is outside the direct file app as external.
    return link.origin !== location.origin;
  } catch (_) {
    return false;
  }
};

export { isValidURL, isExternalLink };
