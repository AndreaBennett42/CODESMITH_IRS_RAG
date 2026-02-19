type OmitNever<T extends Record<string, unknown>> = {
  [K in keyof T as T[K] extends never ? never : K]: T[K];
};

/**
 * Creates a new type of all the shared keys (and their types) between two types
 *
 * e.g.
 * type AB = { a: string, b: number, x: boolean }
 * type BC = { b: number, c: string, x: string }
 * type Common = CommonPropsBetween<AB, BC> = { b: number }
 */
export type CommonPropsBetween<Type1, Type2> = OmitNever<Pick<Type1 & Type2, keyof Type1 & keyof Type2>>;
