import { it, describe, expect } from 'vitest';
import { Path } from '../../flow/Path.js';
import {
  createEnumWrapper,
  createPinWrapper,
  createDollarWrapper,
  createBooleanWrapper,
  createIpPinWrapper,
} from '../persistenceWrappers.js';
import { baseFilerData, mfjFilerData, spouseId, primaryFilerId } from '../testData.js';
import { setupFactGraph } from '../setupFactGraph.js';

const baseData = {
  ...baseFilerData,
  [`/selfSelectPin`]: createPinWrapper(`12349`),
  [`/filedLastYear`]: createBooleanWrapper(true),
};
const baseDataMFJ = {
  ...mfjFilerData,
  [`/selfSelectPin`]: createPinWrapper(`12349`),
  [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
  [`/filedLastYear`]: createBooleanWrapper(true),
  [`/spouseFiledLastYear`]: createBooleanWrapper(true),
};
describe(`for a single filer`, () => {
  /* ------------------------ Selected pins ------------------------ */
  it(`complete when self selected pin and return identity for primary filer is provided`, ({ task }) => {
    task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/selfSelectPin`]: createPinWrapper(`12349`),
      [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
      [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
    });
    expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(true);
  });
  it(`incomplete when self selected pin is NOT provided`, ({ task }) => {
    task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
      [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
    });
    expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(false);
  });
  /* ------------------------ IP Pins ------------------------ */
  it(`complete without a sign identity if primary filer has an IP pin`, ({ task }) => {
    task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];
    const { factGraph } = setupFactGraph({
      ...baseData,
      [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(true),
      [`/filers/#${primaryFilerId}/identityPin`]: createIpPinWrapper(`123456`),
    });
    expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(true);
  });
  /* ------------------------ Did not file taxes last year ------------------------ */
  it(`complete without a sign identity if primary filer did not file taxes last year`, ({ task }) => {
    task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/selfSelectPin`]: createPinWrapper(`12349`),
      [`/filedLastYear`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(true);
  });
  describe(` when sign return identity is`, () => {
    /* ------------------------ Last Year's AGI  ------------------------ */
    describe(`last year's agi`, () => {
      it(`complete when last year agi for primary filer is provided`, ({ task }) => {
        task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];

        const { factGraph } = setupFactGraph({
          ...baseData,
          [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
          [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        });
        expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(true);
      });
      it(`incomplete when last year agi is incomplete`, ({ task }) => {
        task.meta.testedFactPaths = [`/signSectionCompleteSingleFiler`];

        const { factGraph } = setupFactGraph({
          ...baseData,
          [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        });
        expect(factGraph.get(Path.concretePath(`/signSectionCompleteSingleFiler`, null)).get).toBe(false);
      });
    });
  });
  /*
    ------------------ Scenarios for MFJ filers ----------------
   */
  describe(`For MFJ filer`, () => {
    /* ------------------------ Selected Pins ------------------------ */
    it(`complete when self and spouse selected pin and return identity for primary filer and spouse are provided`, ({
      task,
    }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/selfSelectPin`]: createPinWrapper(`12349`),
        [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
        [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
        [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/filedLastYear`]: createBooleanWrapper(true),
        [`/spouseFiledLastYear`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(true);
    });
    it(`incomplete when self selected pin is NOT provided`, ({ task }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
        [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
        [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/filedLastYear`]: createBooleanWrapper(true),
        [`/spouseFiledLastYear`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(false);
    });
    it(`incomplete when spouse selected pin NOT provided`, ({ task }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/selfSelectPin`]: createPinWrapper(`12349`),
        [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
        [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/filedLastYear`]: createBooleanWrapper(true),
        [`/spouseFiledLastYear`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(false);
    });
    it(`incomplete when self selected pin is NOT provided`, ({ task }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
        [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
        [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(false);
    });
    /* ------------------------ IP Pins ------------------------ */
    it(`complete without a sign identity if spouse filer has an IP pin`, ({ task }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/selfSelectPin`]: createPinWrapper(`12349`),
        [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
        [`/filers/#${primaryFilerId}/hasIpPin`]: createBooleanWrapper(true),
        [`/filers/#${primaryFilerId}/identityPin`]: createIpPinWrapper(`123456`),
        [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
        [`/filers/#${spouseId}/identityPin`]: createIpPinWrapper(`123456`),
        [`/filedLastYear`]: createBooleanWrapper(true),
        [`/spouseFiledLastYear`]: createBooleanWrapper(true),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(true);
    });
    /* ------------------------ Did not file taxes last year ------------------------ */
    it(`complete without a sign identity if spouse filer did not file taxes last year`, ({ task }) => {
      task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
      const { factGraph } = setupFactGraph({
        ...mfjFilerData,
        [`/selfSelectPin`]: createPinWrapper(`12349`),
        [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
        [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
        [`/spouseSelfSelectPin`]: createPinWrapper(`12121`),
        [`/filers/#${primaryFilerId}/identityPin`]: createIpPinWrapper(`123456`),
        [`/spouseFiledLastYear`]: createBooleanWrapper(false),
      });
      expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(true);
    });
    /* ------------------------ Return Idenities  ------------------------ */
    describe(` when sign return identity is`, () => {
      /* ------------------------ Last Year's AGI  ------------------------ */
      describe(`last year's agi`, () => {
        it(`complete when last year agi for widowed primary filer is provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseData,
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
            [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            true
          );
        });
        it(`complete when last year agi for primary filer and spouse are provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
            [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
            [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            true
          );
        });
        it(`incomplete when last year agi for primary filer is NOT provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
            [`/spouseLastYearAgi`]: createDollarWrapper(`1000.00`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            false
          );
        });
        it(`incomplete when last year agi for spouse is NOT provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentityOptions`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearAgi`, `/signReturnIdentitySpouseOptions`),
            [`/lastYearAgi`]: createDollarWrapper(`1000.00`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            false
          );
        });
      });
      /* ------------------------ Last Year's Pin  ------------------------ */
      describe(`last year's pin`, () => {
        it(`complete when last year pin for widowed primary filer is provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseData,
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentityOptions`),
            [`/selfSelectPinLastYear`]: createPinWrapper(`11011`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            true
          );
        });
        it(`complete when last year pin for primary filer and spouse are provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentityOptions`),
            [`/selfSelectPinLastYear`]: createPinWrapper(`11011`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentitySpouseOptions`),
            [`/spouseSelfSelectPinLastYear`]: createPinWrapper(`22022`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            true
          );
        });
        it(`incomplete when last year pin for primary filer is NOT provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];

          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentityOptions`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentitySpouseOptions`),
            [`/spouseSelfSelectPinLastYear`]: createPinWrapper(`22022`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            false
          );
        });
        it(`incomplete when last year pin for spouse is NOT provided`, ({ task }) => {
          task.meta.testedFactPaths = [`/signSectionCompleteMarriedFilingJointlyFilers`];
          const { factGraph } = setupFactGraph({
            ...baseDataMFJ,
            [`/signReturnIdentity`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentityOptions`),
            [`/signReturnIdentitySpouse`]: createEnumWrapper(`lastYearPin`, `/signReturnIdentitySpouseOptions`),
            [`/selfSelectPinLastYear`]: createPinWrapper(`11011`),
          });
          expect(factGraph.get(Path.concretePath(`/signSectionCompleteMarriedFilingJointlyFilers`, null)).get).toBe(
            false
          );
        });
      });
    });
  });
});
