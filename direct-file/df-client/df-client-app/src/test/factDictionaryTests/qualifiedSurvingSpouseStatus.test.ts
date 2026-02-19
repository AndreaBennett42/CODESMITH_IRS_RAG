import { describe, it, expect } from 'vitest';
import { createBooleanWrapper, createEnumWrapper } from '../persistenceWrappers.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';

describe(`Qualified surviving spouse`, () => {
  it(`Is ineligible for QSS if spouse died in the tax year`, ({ task }) => {
    task.meta.testedFactPaths = [`/eligibleForQss`];
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
    });
    const isEligibleForQss = factGraph.get(`/eligibleForQss` as ConcretePath);
    expect(isEligibleForQss.complete).toBe(true);
    expect(isEligibleForQss.get).toBe(false);
  });
  it(`Is ineligible for QSS if spouse died 3 or more years before the tax year`, ({ task }) => {
    task.meta.testedFactPaths = [`/eligibleForQss`];

    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`beforeTaxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
    });
    const isEligibleForQss = factGraph.get(`/eligibleForQss` as ConcretePath);
    expect(isEligibleForQss.complete).toBe(true);
    expect(isEligibleForQss.get).toBe(false);
  });
  it(`Is ineligible for QSS if TP weren't able to file jointly the year of the spouse's death`, ({ task }) => {
    task.meta.testedFactPaths = [`/eligibleForQss`];

    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(false),
    });
    const isEligibleForQss = factGraph.get(`/eligibleForQss` as ConcretePath);
    expect(isEligibleForQss.complete).toBe(true);
    expect(isEligibleForQss.get).toBe(false);
  });

  it.skip(`Is only eligible when a dependent qualifies you as QSS`, () => {
    // TODO: this will come after redoing the family + household section
    // since I don't want to rewrite /qssQualifyingPerson logic more times than I need to
    const { factGraph } = setupFactGraph({
      [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
    });
    const isEligibleForQss = factGraph.get(`/eligibleForQss` as ConcretePath);
    expect(isEligibleForQss.complete).toBe(true);
    expect(isEligibleForQss.get).toBe(true);
  });
});
