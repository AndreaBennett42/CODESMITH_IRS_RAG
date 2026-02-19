import { it, describe, expect } from 'vitest';
import { Path } from '../../flow/Path.js';
import { createCollectionWrapper, createEnumWrapper, createTinWrapper } from '../persistenceWrappers.js';
import { baseFilerData, primaryFilerId, spouseId, singleFilerData } from '../testData.js';
import { scalaListToJsArray } from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';

const twoFilers = {
  '/filers': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: { items: [primaryFilerId, spouseId] },
  },
};

describe(`/filers/*/isTinUnique`, () => {
  it(`should return true if married and filers have unique tins`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/isTinUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    });
    expect(factGraph.get(Path.concretePath(`/filersWithTins`, null)).complete).toBe(true);
    const results = factGraph.get(Path.concretePath(`/filersWithTins`, null)).get.getItemsAsStrings();
    const resultArray = scalaListToJsArray(results);
    expect(resultArray.length).toEqual(2);

    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
  });

  it(`should return false if married and filers have matching tins`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/isTinUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    });
    expect(factGraph.get(Path.concretePath(`/filersWithTins`, null)).complete).toBe(true);
    const results = factGraph.get(Path.concretePath(`/filersWithTins`, null)).get.getItemsAsStrings();
    const resultArray = scalaListToJsArray(results);
    expect(resultArray.length).toEqual(2);

    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(false);
  });

  it(`should return true if single`, ({ task }) => {
    task.meta.testedFactPaths = [`/filers/*/isTinUnique`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    });
    expect(factGraph.get(Path.concretePath(`/filersWithTins`, null)).complete).toBe(true);
    const results = factGraph.get(Path.concretePath(`/filersWithTins`, null)).get.getItemsAsStrings();
    const resultArray = scalaListToJsArray(results);
    expect(resultArray.length).toEqual(1);

    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
  });
});

describe(`/allTinsUnique`, () => {
  const child1DependentId = `4fa3a5a7-a9d1-43a9-a0fb-277596e70d48`;
  const child2DependentId = `a39ce7e2-3d09-11ee-be56-0242ac120002`;

  it(`should return true,  if single filer with no dependents`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      '/familyAndHousehold': createCollectionWrapper([]),
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // AllTinsUnique returns true
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(true);
  });

  it(`should return false, if single filer, dependent tin matches filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),

      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // Dependents are not unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).get).toBe(false);
    // AllTinsUnique returns false
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(false);
  });

  it(`should return true,  if single filer, one dependent with tin, one without`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),

      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // Dependents are unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).get).toBe(true);
    // AllTinsUnique returns false
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(true);
  });

  it(`should return true,  if single filer, dependent tin doesn't match filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...singleFilerData,
      [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
      [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),

      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5559` }),
    });

    // AllTinsUnique returns true
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(true);
  });

  it(`should return true,  if married with no dependents`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      '/familyAndHousehold': createCollectionWrapper([]),
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5557` }),
    });

    // AllTinsUnique returns true
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(true);
  });

  it(`should return true,  if married with dependents, all unique tins`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5557` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5559` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // Dependents are unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).get).toBe(true);
    // AllTinsUnique returns true
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(true);
  });

  it(`should return false, if married with dependents, filer tins non-unique`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5559` }),
    });

    // Filers are not unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(false);
    // Dependents are unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child1DependentId)).get).toBe(true);
    // AllTinsUnique returns false
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(false);
  });

  it(`should return false, if married with dependents, dependent tins non-unique`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // Dependents are not unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).get).toBe(false);
    // AllTinsUnique returns false
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(false);
  });

  it(`should return false, if married with dependents, dependent tin matches filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/allTinsUnique`];

    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      ...twoFilers,
      '/familyAndHousehold': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${child1DependentId}`, `${child2DependentId}`] },
      },

      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
      [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
      [`/familyAndHousehold/#${child1DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5558` }),
      [`/familyAndHousehold/#${child2DependentId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5556` }),
    });

    // Filers are unique
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/filers/*/isTinUnique`, primaryFilerId)).get).toBe(true);
    // Dependents are not unique
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).complete).toBe(
      true
    );
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/isTinUnique`, child2DependentId)).get).toBe(false);
    // AllTinsUnique returns false
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).complete).toBe(true);
    expect(factGraph.get(Path.concretePath(`/allTinsUnique`, null)).get).toBe(false);
  });
});
