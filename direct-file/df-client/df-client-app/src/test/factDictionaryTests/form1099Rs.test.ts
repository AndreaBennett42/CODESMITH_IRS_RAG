import { Path } from '../../flow/Path.js';
import {
  createCollectionWrapper,
  createDollarWrapper,
  createEnumWrapper,
  createStringWrapper,
} from '../persistenceWrappers.js';
import { setupFactGraph } from '../setupFactGraph.js';
import { baseFilerData, primaryFilerId, spouseId } from '../testData.js';
import { describe, it, expect } from 'vitest';

describe(`1099-R nonstandardOrCorrectedChoice determines standardOrNonStandardCd`, () => {
  it(`when nonstandardOrCorrectedChoice is marked neither, standardOrNonStandardCd is 'S'`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099Rs/*/standardOrNonStandardCd`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Rs/#${primaryFilerId}/nonstandardOrCorrectedChoice`]: createEnumWrapper(
        `neither`,
        `/form1099RsNonstandardCorrectedOptions`
      ),
    });

    const standardOrNonStandardCd = factGraph.get(
      Path.concretePath(`/form1099Rs/*/standardOrNonStandardCd`, primaryFilerId)
    );
    expect(standardOrNonStandardCd.get.toString()).toBe(`S`);
  });

  it(`when nonstandardOrCorrectedChoice is marked nonstandard, standardOrNonStandardCd is 'N'`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099Rs/*/standardOrNonStandardCd`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Rs/#${primaryFilerId}/nonstandardOrCorrectedChoice`]: createEnumWrapper(
        `nonstandard`,
        `/form1099RsNonstandardCorrectedOptions`
      ),
    });

    const isNonStandard = factGraph.get(Path.concretePath(`/form1099Rs/*/isNonStandard`, primaryFilerId));
    expect(isNonStandard.get.toString()).toBe(`true`);

    const standardOrNonStandardCd = factGraph.get(
      Path.concretePath(`/form1099Rs/*/standardOrNonStandardCd`, primaryFilerId)
    );
    expect(standardOrNonStandardCd.get.toString()).toBe(`N`);
  });

  it(`when nonstandardOrCorrectedChoice is marked both, standardOrNonStandardCd is 'N'`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099Rs/*/standardOrNonStandardCd`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Rs/#${primaryFilerId}/nonstandardOrCorrectedChoice`]: createEnumWrapper(
        `both`,
        `/form1099RsNonstandardCorrectedOptions`
      ),
    });

    const standardOrNonStandardCd = factGraph.get(
      Path.concretePath(`/form1099Rs/*/standardOrNonStandardCd`, primaryFilerId)
    );
    expect(standardOrNonStandardCd.get.toString()).toBe(`N`);
  });
});

describe(`1099-R nonstandardOrCorrectedChoice determines isCorrected`, () => {
  it(`when nonstandardOrCorrectedChoice is marked corrected, isCorrected is 'true'`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099Rs/*/isCorrected`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Rs/#${primaryFilerId}/nonstandardOrCorrectedChoice`]: createEnumWrapper(
        `corrected`,
        `/form1099RsNonstandardCorrectedOptions`
      ),
    });

    const isNonStandard = factGraph.get(Path.concretePath(`/form1099Rs/*/isCorrected`, primaryFilerId));
    expect(isNonStandard.get.toString()).toBe(`true`);
  });
});

describe(`1099-R distribution code(s)`, () => {
  const supportedCodes = [`2`, `7`, `G`, `H`, `2B`, `B2`, `7B`, `B7`, `GB`, `BG`, `4G`, `G4`, `H4`, `4H`];

  for (const code of supportedCodes) {
    it(`should support ` + code, ({ task }) => {
      task.meta.testedFactPaths = [`/form1099Rs/*/writableDistributionCode`];
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
        [`/form1099Rs/#${primaryFilerId}/writableDistributionCode`]: createStringWrapper(code),
      });

      expect(
        factGraph.get(Path.concretePath(`/form1099Rs/*/hasUnsupportedDistributionCode`, primaryFilerId)).get.toString()
      ).toBe(`false`);
    });
  }

  const unsupportedCodes = [
    `0`,
    `5`,
    `6`,
    `8`,
    `9`,
    `A`,
    `C`,
    `D`,
    `e`,
    `123`,
    ``,
    `24`,
    `27`,
    `2G`,
    `2H`,
    `47`,
    `4B`,
    `7G`,
    `7H`,
    `GH`,
    `BH`,
  ];
  // Note: double valid characters such as `11` are also unsupported, but that cannot be tested here.
  // It is enforced by limit regex on the input component.

  for (const code of unsupportedCodes) {
    it(`should not support ` + code, ({ task }) => {
      task.meta.testedFactPaths = [`/form1099Rs/*/writableDistributionCode`];
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
        [`/form1099Rs/#${primaryFilerId}/writableDistributionCode`]: createStringWrapper(code),
      });

      expect(
        factGraph.get(Path.concretePath(`/form1099Rs/*/hasUnsupportedDistributionCode`, primaryFilerId)).get.toString()
      ).toBe(`true`);
    });
  }
});

describe(`1099-R taxes withheld`, () => {
  describe(`/form1099Rs/*/hasMoreTaxWithheldThanDistributions`, () => {
    it.each([
      [`1000.00`, undefined, undefined, false, false],
      [`1000.00`, `500.00`, undefined, false, false],
      [`1000.00`, `500.00`, `500.00`, false, false],
      [`1000.00`, `1500.00`, undefined, true, true],
      [`1000.00`, `1000.00`, `500.00`, true, true],
      [`1000.00`, undefined, `1001.00`, true, true],
    ])(
      `When grossDistribution is %s, stateTaxWithheld is %s, and localTaxWithheld is %s, 
      hasMoreTaxWithheldThanDistributions is %s and hasMoreTaxThanDistributionsOn1099Rs is %s`,
      (
        grossDistribution,
        stateTaxWithheld,
        localTaxWithheld,
        hasMoreTaxWithheldThanDistributions,
        hasMoreTaxThanDistributionsOn1099Rs
      ) => {
        const { factGraph } = setupFactGraph({
          ...baseFilerData,
          [`/form1099Rs`]: createCollectionWrapper([primaryFilerId, spouseId]),
          [`/form1099Rs/#${primaryFilerId}/writableGrossDistribution`]: createDollarWrapper(grossDistribution),
          [`/form1099Rs/#${primaryFilerId}/writableStateTaxWithheld`]: stateTaxWithheld
            ? createDollarWrapper(stateTaxWithheld)
            : undefined,
          [`/form1099Rs/#${primaryFilerId}/writableLocalTaxWithheld`]: localTaxWithheld
            ? createDollarWrapper(localTaxWithheld)
            : undefined,
        });

        expect(
          factGraph
            .get(Path.concretePath(`/form1099Rs/*/hasMoreTaxWithheldThanDistributions`, primaryFilerId))
            .get.toString()
        ).toBe(hasMoreTaxWithheldThanDistributions.toString());
        expect(
          factGraph.get(Path.concretePath(`/hasMoreTaxThanDistributionsOn1099Rs`, primaryFilerId)).get.toString()
        ).toBe(hasMoreTaxThanDistributionsOn1099Rs.toString());
      }
    );
  });
});
