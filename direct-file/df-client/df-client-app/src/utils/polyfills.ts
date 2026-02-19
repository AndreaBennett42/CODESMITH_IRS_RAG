/*
 * Polyfills for functions that we can't use because they are unsupported on older browsers.
 */

export function findLast<T>(arr: T[], func: (item: T) => boolean): T | undefined {
  for (let i = arr.length - 1; i >= 0; i--) {
    const value = arr[i];
    if (func(value)) {
      return value;
    }
  }

  // Technically not necessary, but I'm being explicit
  return undefined;
}

export function hasOwn(obj: object | undefined, prop: string): boolean {
  if (!obj) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(obj, prop);
}
