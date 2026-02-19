import { Path } from './Path.js';

describe(`Path`, () => {
  it(`can make a concete path without a collectionId`, () => {
    const samplePath = `/taxYear`;
    expect(Path.concretePath(samplePath, null)).toEqual(samplePath);
  });
  it(`can make a concrete path with a collectionId`, () => {
    const samplePath = `/filers/*/firstName`;
    const collectionId = `9bb00875-1614-45c7-b71c-a2474a76917e`;
    expect(Path.concretePath(samplePath, collectionId)).toEqual(`/filers/#${collectionId}/firstName`);
  });

  it(`ignores the collectionId where appropriate`, () => {
    const samplePath = `/address`;
    const collectionId = `9bb00875-1614-45c7-b71c-a2474a76917e`;
    expect(Path.concretePath(samplePath, collectionId)).toEqual(samplePath);
  });
});
