/**
 * Typescript enums have lots of duplication. When the entire point is "I want to write a string once and only once",
 * this utility can be handy
 * @param enumMembers array of enum keys
 * @return an enum-like object, with all the enumMembers accessible
 */
export const createStringEnum = <T extends string>(enumMembers: T[]): { [k in T]: k } => {
  return enumMembers.reduce((acc, item) => {
    acc[item] = item;
    return acc;
  }, {} as { [k in T]: k });
};
