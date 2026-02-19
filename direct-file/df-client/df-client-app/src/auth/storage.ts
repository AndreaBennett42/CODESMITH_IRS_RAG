
export const clearBrowserStorage = () => {
  // Local storage is a per-origin data cache in the user's browser.
  localStorage.clear();
  // Session storage is similar, but limited to a single tab/page.
  sessionStorage.clear();
};

export const initPrivacyHandler = () => {
  document.addEventListener(`click`, (e) => {
    const target = ((e.target as HTMLElement) || null)?.closest(`a`);
    const href = target?.getAttribute(`href`) || ``;
    const isLogout = href.indexOf(import.meta.env.VITE_SADI_LOGOUT_URL) !== -1;
    if (isLogout) clearBrowserStorage();
  });
};
