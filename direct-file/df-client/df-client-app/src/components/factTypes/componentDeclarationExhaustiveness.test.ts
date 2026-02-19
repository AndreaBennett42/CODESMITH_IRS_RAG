import { FactTypeRenderer, CollectionItemManagerRenderer, InfoTypeRenderer } from './index.js';
import { FactTypes, InfoDisplayTypes, CollectionItemManagerTypes } from '../../flow/ContentDeclarations.js';
describe(`Screen Content Node declarations have renderers`, () => {
  it(`Every fact type declaration has a fact type renderer`, () => {
    expect(Object.keys(FactTypeRenderer).sort()).toEqual(Object.keys(FactTypes).sort());
  });
  it(`Every infodisplay declaration has a fact type renderer`, () => {
    expect(Object.keys(InfoTypeRenderer).sort()).toEqual(Object.keys(InfoDisplayTypes).sort());
  });
  it(`Every collectionItemManager declaration has a fact type renderer`, () => {
    expect(Object.keys(CollectionItemManagerRenderer).sort()).toEqual(Object.keys(CollectionItemManagerTypes).sort());
  });
});
