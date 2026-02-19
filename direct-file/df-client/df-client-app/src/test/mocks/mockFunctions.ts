import mockEnYaml from '../../locales/en.yaml';

export const mockUseTranslation = (_ns?: string, options?: { keyPrefix: string }) => {
  // The t function is given the path in the en.yaml file
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const t = (key: any): string => {
    const path = options?.keyPrefix ? `${options.keyPrefix}.${key}` : key;
    if (Array.isArray(path)) return path.reduce((result, p) => result || t(p), ``);
    return (
      path
        // split the path into an array
        .split(`.`)
        /**
         * The reduce method traverses the object's hierarchy and returns
         * the value of the nested property. The first arg (currentObj) is
         * initialized to enYaml (adding a "mock" prefix is required as
         * without it, it's out of scope). In each iteration, the currentObj
         * is overwritten with the current key. When it's done traversing,
         * we have reached the end of the path and the value is returned.
         */
        .reduce(
          (currentObj: { [x: string]: string }, key: string | number) => currentObj && currentObj[key],
          mockEnYaml
        )
    );
  };
  const i18n = {
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeLanguage: () => new Promise(() => {}),
    exists: (path: string) => !!t(path), // let t figure out if the path exists
  };

  return { t, i18n };
};

// session storage mocks
export const sessionStorageMock = (() => {
  type Key = string;
  type Value = { toString: () => string };
  let store: Record<Key, Value> = {};
  return {
    getItem: (key: Key) => store[key],
    setItem: (key: Key, value: Value) => (store[key] = value.toString()),
    clear: () => (store = {}),
    removeItem: (key: Key) => delete store[key],
  };
})();
