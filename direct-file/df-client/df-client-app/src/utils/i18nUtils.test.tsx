import { getTranslationValuesRecursive } from './i18nUtils.js';

describe(getTranslationValuesRecursive.name, () => {
  it(`Returns an array with just the value for string values`, () => {
    const value = `a string`;
    const values = getTranslationValuesRecursive(value);

    expect(values.length).toEqual(1);
    expect(values.includes(value)).toBeTruthy();
  });

  it(`Returns an array of all the values in an array`, () => {
    const value = [`string 1`, `string 2`, `string 3`];
    const values = getTranslationValuesRecursive(value);

    expect(values.length).toEqual(3);
    value.forEach((arrayChild) => {
      expect(values.includes(arrayChild)).toBeTruthy();
    });
  });

  it(`Returns an array of all the values in an object`, () => {
    const value = {
      key1: `string 1`,
      key2: `string 2`,
      key3: `string 3`,
    };
    const values = getTranslationValuesRecursive(value);

    expect(values.length).toEqual(3);
    Object.values(value).forEach((objectValue) => {
      expect(values.includes(objectValue)).toBeTruthy();
    });
  });

  it(`Returns all the values of a complex array`, () => {
    const nestedArray = [`nested value 1`, `nested value 2`, `nested value 3`];
    const objectNestedArray = [`object nested string 1`, `object nested string 2`, `object nested string 3`];
    const value = [
      `value 1`,
      nestedArray,
      {
        nestedKey1: `nested value 1`,
        nestedKey2: `nested value 2`,
        nestedKey3: objectNestedArray,
      },
    ];
    const values = getTranslationValuesRecursive(value);

    expect(values.length).toEqual(3 + nestedArray.length + objectNestedArray.length);

    // includes the string values
    expect(values.includes(`value 1`)).toBeTruthy();
    expect(values.includes(`nested value 1`)).toBeTruthy();
    expect(values.includes(`nested value 2`)).toBeTruthy();

    // includes the nested array values
    nestedArray.forEach((arrayChild) => {
      expect(values.includes(arrayChild)).toBeTruthy();
    });

    // includes the nested object values
    Object.values(objectNestedArray).forEach((objectValue) => {
      expect(values.includes(objectValue)).toBeTruthy();
    });
  });

  it(`Returns all the values of a complex object`, () => {
    const nestedArray = [`nested value 1`, `nested value 2`, `nested value 3`];
    const objectNestedArray = [`object nested string 1`, `object nested string 2`, `object nested string 3`];
    const value = {
      key1: `value 1`,
      key2: nestedArray,
      key3: {
        nestedKey1: `nested value 1`,
        nestedKey2: `nested value 2`,
        nestedKey3: objectNestedArray,
      },
    };
    const values = getTranslationValuesRecursive(value);

    expect(values.length).toEqual(3 + nestedArray.length + objectNestedArray.length);

    // includes the string values
    expect(values.includes(value.key1)).toBeTruthy();
    expect(values.includes(value.key3.nestedKey1)).toBeTruthy();
    expect(values.includes(value.key3.nestedKey2)).toBeTruthy();

    // includes the nested array values
    nestedArray.forEach((arrayChild) => {
      expect(values.includes(arrayChild)).toBeTruthy();
    });

    // includes the nested object values
    Object.values(objectNestedArray).forEach((objectValue) => {
      expect(values.includes(objectValue)).toBeTruthy();
    });
  });
});
