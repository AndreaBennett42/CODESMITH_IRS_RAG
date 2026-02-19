import { findLast, hasOwn } from './polyfills.js';

describe(`findLast`, () => {
  it(`it finds the the last matching element`, () => {
    const arr = [1, 2, 3, 4, 5];
    const res = findLast(arr, (x) => x % 2 === 0);
    expect(res).toEqual(4);
  });

  it(`it returns undefined if none is found`, () => {
    const arr = [1, 2, 3, 4, 5];
    const res = findLast(arr, (x) => x % 2 === 7);
    expect(res).toEqual(undefined);
  });

  it(`it returns undefined on empty arrays`, () => {
    const arr: number[] = [];
    const res = findLast(arr, (x) => x % 2 === 7);
    expect(res).toEqual(undefined);
  });
});

describe(`hasOwn`, () => {
  it(`returns true if the object has that property`, () => {
    const obj = { testprop: 1 };
    const res = hasOwn(obj, `testprop`);
    expect(res).toEqual(true);
  });

  it(`returns false if the object does not have that property`, () => {
    const obj = { testprop: 1 };
    const res = hasOwn(obj, `notaprop`);
    expect(res).toEqual(false);
  });

  it(`returns false if the object is undefined`, () => {
    const obj = undefined;
    const res = hasOwn(obj, `testprop`);
    expect(res).toEqual(false);
  });
});
