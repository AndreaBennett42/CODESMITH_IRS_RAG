import { it, describe, expect } from 'vitest';
import { baseFilerData, primaryFilerId, spouseId } from '../testData.js';
import {
  createBooleanWrapper,
  createDayWrapper,
  createEnumWrapper,
  createIpPinWrapper,
  createMultEnumWrapper,
  createTinWrapper,
} from '../persistenceWrappers.js';
import { Path } from '../../flow/Path.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { EnumFactory } from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);
const modifiedBaseFilerData = {
  [`/filingStatus`]: undefined,
  ...(baseFilerData as Partial<typeof baseFilerData>),
  [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
  [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingJointly`, `/filingStatusInitialOptions`),
};

delete modifiedBaseFilerData[`/filingStatus`];
const stateVariations = [`nv`, `wa`, `ca`];

describe(`The "spouse" subsection`, () => {
  describe(`The "widowed" path`, () => {
    describe(`asks if the filer was in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(` if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner`);
        });
      });
    });

    describe(`asks what year the spouse died if they were not in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(`and if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(false),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-year-of-death`);
        });
      });
    });

    describe(`knocks out of the filer if they were in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(`and if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(true),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner-ko`);
        });
      });
    });

    it(`asks what year your spouse died if the filer selects widowed as martial status`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, spouseId, task)
      ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-year-of-death`);
    });

    it(`skips to filing status if the spouse died TY-3+`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`beforeTaxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-year-of-death`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
    });

    it(`asks about previous filing status for TY-2`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-year-of-death`, spouseId, task)
      ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`);
    });

    it(`asks about previous filing status for TY-1`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-year-of-death`, spouseId, task)
      ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`);
    });

    it(`moves to the outro for TY-2`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusTwo`, `/yearOfSpouseDeathOptions`),
        [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
      factGraph.set(Path.concretePath(`/canFileJointlyYearOfSpouseDeath`, spouseId), false);
      factGraph.save();
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
    });

    it(`moves to the outro for TY-1`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYearMinusOne`, `/yearOfSpouseDeathOptions`),
        [`/canFileJointlyYearOfSpouseDeath`]: createBooleanWrapper(true),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
      factGraph.set(Path.concretePath(`/canFileJointlyYearOfSpouseDeath`, spouseId), false);
      factGraph.save();
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-widowed-entitled-mfj`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
    });
    it(`moves into the Married Filing Jointly flow for TY`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
        [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-year-of-death`, spouseId, task)
      ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-live-together`);
    });
  });
});

describe(`The married path`, () => {
  it(`moves to the married filing jointly flow for married filers`, ({ task }) => {
    const { factGraph } = setupFactGraph({
      ...modifiedBaseFilerData,
      [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
    });
    expect(
      givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, spouseId, task)
    ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-live-together`);
  });

  describe(`The Married Filing Jointly/Separately flows`, () => {
    // The two statuses that are eligible for MFJ/MFS
    // Married, and Widowed this year
    const maritalStatusVariations = [
      {
        name: `married`,
        facts: {
          [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
        },
      },
      {
        name: `widowed`,
        facts: {
          [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
          [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
        },
      },
    ];
    const twoFilers = {
      [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
      [`/filers/#${spouseId}/isBlind`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(false),
      [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
      [`/spouseLivesInTPState`]: createEnumWrapper(`sameState`, `/spouseScopedStateOptions`),

      [`/spouseW2And1099IntInScopedState`]: createEnumWrapper(`noForms`, `/spouseW2And1099IntStateOptions`),
      [`/primaryFilerW2And1099IntInScopedState`]: createEnumWrapper(`noForms`, `/primaryFilerW2And1099IntStateOptions`),
    };
    describe(`The cohabitation flow`, () => {
      it(
        `asks filer how many months they lived together` + ` if they were living apart in the tax year`,
        ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-live-together`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-live-together-duration`);
          }
        }
      );
      it(
        `asks filer if they lived apart the last six months of the year` +
          ` if they were living together half the year or less`,
        ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherSixMonthsOrLess`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              ...maritalStatus.facts,
            });

            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-live-together-duration`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-apart-last-6-mo`);

            factGraph.set(
              Path.concretePath(`/spouseLivedTogetherMonths`, spouseId),
              EnumFactory(`livedTogetherMoreThanSixMonths`, `/spouseLivedTogetherMonthsOptions`).right
            );
            factGraph.save();

            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-live-together-duration`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`);
          }
        }
      );
      it(`asks if they had a separation agreement` + ` if they were living apart the last six months`, ({ task }) => {
        for (const maritalStatus of maritalStatusVariations) {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
            [`/spouseLivedTogetherMonths`]: createEnumWrapper(
              `livedTogetherSixMonthsOrLess`,
              `/spouseLivedTogetherMonthsOptions`
            ),
            [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(false),
            ...maritalStatus.facts,
          });

          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-apart-last-6-mo`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`);
        }
      });
      it(
        `asks if the filer had a separation agreement` + ` if they were living together more than half the year`,
        ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherMoreThanSixMonths`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-live-together-duration`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`);
          }
        }
      );

      describe(`moves to ask about the spouse's citizenship status`, () => {
        it(`after answering if the filer had a separation agreement`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherMoreThanSixMonths`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
              [`/writableSeparationAgreement`]: createBooleanWrapper(true),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-citizenship`);

            factGraph.set(Path.concretePath(`/writableSeparationAgreement`, spouseId), false);
            factGraph.save();

            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-citizenship`);
          }
        });

        it(`regardless of if the filer had a separation agreement`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherMoreThanSixMonths`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              [`/writableSeparationAgreement`]: createBooleanWrapper(true),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-citizenship`);

            factGraph.set(Path.concretePath(`/writableSeparationAgreement`, spouseId), false);
            factGraph.save();
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-living-apart-sep-agreement-last-day`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-citizenship`);
          }
        });
      });

      describe(`moves to choose the correct filing status choice screen`, () => {
        it(
          `routes to filing status choice a if ` +
            `the marital status is married or widowed (and spouse died in tax year) and ` +
            `both are citizens / residents all year or citizens end of year or residents for tax purposes, and ` +
            `the filer has lived apart for the last 6 months of the year`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherSixMonthsOrLess`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-citizenship`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/filing-status-choice-a`);
          }
        );

        it(
          `routes to filing status choice b if ` +
            `the marital status is married or widowed (and spouse died in tax year) and ` +
            `both are citizens / residents all year or citizens end of year or residents for tax purposes, and ` +
            `the filer has not lived apart for the last 6 months of the year`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/writableSeparationAgreement`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-citizenship`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/filing-status-choice-b`);
          }
        );
      });

      describe(`moves to the correct screen depending on what filing choice was picked`, () => {
        it(`moves to the spouse outro if the head of household choice was picked`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
            [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
            [`/spouseLivedTogetherMonths`]: createEnumWrapper(
              `livedTogetherSixMonthsOrLess`,
              `/spouseLivedTogetherMonthsOptions`
            ),
            [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
            [`/filingStatusChoice`]: createEnumWrapper(
              `wantsToSeeIfTheyQualifyHeadOfHousehold`,
              `/filingStatusInitialOptions`
            ),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-a`, spouseId, task)
          ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
        });

        describe(`routes to add spouse a if the filer has picked MFJ regardless of`, () => {
          it(`if the filer came from FilingStatusChoiceA`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherSixMonthsOrLess`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-a`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/add-spouse-a`);
          });

          it(`if the filer came from FilingStatusChoiceB`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/writableSeparationAgreement`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-b`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/add-spouse-b`);
          });
        });

        describe(`routes to add spouse b if the filer has picked MFS regardless of`, () => {
          it(`if the filer came from FilingStatusChoiceA`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(false),
              [`/writableLivedApartLastSixMonths`]: createBooleanWrapper(true),
              [`/spouseLivedTogetherMonths`]: createEnumWrapper(
                `livedTogetherSixMonthsOrLess`,
                `/spouseLivedTogetherMonthsOptions`
              ),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-a`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/add-spouse-a`);
          });

          it(`if the filer came from FilingStatusChoiceB`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/writableSeparationAgreement`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-b`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/add-spouse-b`);
          });
        });
      });

      describe(`Married filing separately`, () => {
        it(`moves to the MFS spouse intro if the filer is adding a spouse to the MFS section`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
            [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
            [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/add-spouse-b`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-data-intro`);
        });

        it(`moves to ask if the spouse is filing a return if the filer is continuing from the MFS data intro`, ({
          task,
        }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
            [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
            [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-data-intro`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-living`);
        });

        it(
          `moves to ask if the spouse will have a return filed for them if the filer ` + `selects MFS and is widowed`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-data-intro`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-widowed`);
          }
        );

        it(`moves to ask if the spouse will be itemizing if the spouse will file a return`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
            [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
            [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
            [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
          });
          expect(
            givenFacts(factGraph).atPath(
              `/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-living`,
              spouseId,
              task
            )
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-itemizing`);
        });

        it(
          `moves to ask if the spouse will be itemizing` + ` if the spouse will have a return filed for them`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
              [`/MFSDeceasedSpouseFilingReturn`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-widowed`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-itemizing`);
          }
        );

        it(`knocks the filer out if their spouse itemizes`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const MFSSpouseFilingFact =
              maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [MFSSpouseFilingFact]: createBooleanWrapper(true),
              [`/spouseItemizes`]: createBooleanWrapper(true),
              ...maritalStatus.facts,
            });

            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-itemizing`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-itemizing-ko`);
          }
        });

        describe(`moves to ask about the spouse's name`, () => {
          const expectedNextScreen = `/flow/you-and-your-family/spouse/mfs-spouse-name` + ``;

          it(`if the spouse is not itemizing`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
              [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
              [`/spouseItemizes`]: createBooleanWrapper(false),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-itemizing`, spouseId, task)
            ).toRouteNextTo(expectedNextScreen);
          });

          it(`if the spouse will not have a return filed for them`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-living`,
                spouseId,
                task
              )
            ).toRouteNextTo(expectedNextScreen);
          });

          it(`if the spouse is not filing a return`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
              [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/mfs-spouse-filing-a-return-widowed`,
                spouseId,
                task
              )
            ).toRouteNextTo(expectedNextScreen);
          });
        });

        it(`moves to ask about the spouse's NRTIN if the spouse is a not a US citizen`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const MFSSpouseFilingFact =
              maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [MFSSpouseFilingFact]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-name`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-nr-tin`);
          }
        });

        describe(`moves to ask about the spouse's tax id`, () => {
          it(`if the spouse is a US citizen`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const MFSSpouseFilingFact =
                maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [MFSSpouseFilingFact]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-name`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-tax-id`);
            }
          });

          it(`if the spouse has a NRTIN`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const MFSSpouseFilingFact =
                maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [MFSSpouseFilingFact]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
                [`/MFSSpouseHasNRTIN`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-nr-tin`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-tax-id`);
            }
          });
        });

        describe(`moves to ask about the spouse's gross income`, () => {
          it(`if the spouse does not have a NRTIN`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const MFSSpouseFilingFact =
                maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [MFSSpouseFilingFact]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
                [`/MFSSpouseHasNRTIN`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-nr-tin`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-gross-income`);
            }
          });

          it(`if the spouse is not filing a tax return and has entered in a tax id`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const MFSSpouseFilingFact =
                maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [MFSSpouseFilingFact]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
                [`/MFSSpouseHasNRTIN`]: createBooleanWrapper(true),
                [Path.concretePath(`/filers/*/tin`, spouseId)]: createTinWrapper({
                  area: `222`,
                  group: `12`,
                  serial: `1234`,
                }),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-tax-id`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-gross-income`);
            }
          });
        });

        it(`moves to ask if spouse could be claimed as a dependent if the spouse had no gross income`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const MFSSpouseFilingFact =
              maritalStatus.name === `married` ? `/MFSLivingSpouseFilingReturn` : `/MFSDeceasedSpouseFilingReturn`;

            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [MFSSpouseFilingFact]: createBooleanWrapper(false),
              [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-gross-income`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfs-spouse-could-be-claimed`);
          }
        });

        describe(`moves to the spouse blind page regardless of`, () => {
          const expectedNextScreen = `/flow/you-and-your-family/spouse/mfs-spouse-blind` + ``;

          it(`if the spouse has reached age 65 or not`, ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
              [`/filingStatusChoice`]: createEnumWrapper(`wantsMarriedFilingSeparately`, `/filingStatusInitialOptions`),
              [`/maritalStatus`]: createEnumWrapper(`married`, `/maritalStatusOptions`),
              [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
              [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
              [`/MFSSpouse65OrOlder`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-age`, spouseId, task)
            ).toRouteNextTo(expectedNextScreen);

            factGraph.set(Path.concretePath(`/MFSSpouse65OrOlder`, spouseId), false);
            factGraph.save();

            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-age`, spouseId, task)
            ).toRouteNextTo(expectedNextScreen);
          });
        });

        describe(`moves to the outro screen`, () => {
          const expectedNextScreen = `/data-view/flow/you-and-your-family/spouse`;

          it(`if the spouse is filing a return`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),

                [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-tax-id`, spouseId, task)
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(`if the spouse has gross income`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,

                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
                [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-gross-income`, spouseId, task)
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(`if the spouse could be claimed as a dependent`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
                [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/mfs-spouse-could-be-claimed`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(`regardless if the spouse is blind or not`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filingStatusChoice`]: createEnumWrapper(
                  `wantsMarriedFilingSeparately`,
                  `/filingStatusInitialOptions`
                ),
                [`/MFSLivingSpouseFilingReturn`]: createBooleanWrapper(false),
                [`/MFSSpouseHasGrossIncome`]: createBooleanWrapper(false),
                [`/filers/#${spouseId}/isBlind`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-blind`, spouseId, task)
              ).toRouteNextTo(expectedNextScreen);

              factGraph.set(Path.concretePath(`/secondaryFiler/isBlind`, spouseId), true);
              factGraph.save();

              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfs-spouse-blind`, spouseId, task)
              ).toRouteNextTo(expectedNextScreen);
            }
          });
        });
      });

      describe(`Married filing jointly`, () => {
        it(`moves to residency scope if the user selects MFJ`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/filing-status-choice-a`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/add-spouse-a`);
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/add-spouse-a`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-state-residency-scope`);
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-intro`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`);
          }
        });

        it(`all inputs on state income form go to APF`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
              [`/spouseW2And1099IntInScopedState`]: createEnumWrapper(`onlySame`, `/spouseW2And1099IntStateOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-state-income-form`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-pfd-income`);
          }
        });

        it(`goes from APF to knockout if spouse has W2 or 1099-INT income for another state`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
              [`/spouseW2And1099IntInScopedState`]: createEnumWrapper(
                `anotherState`,
                `/spouseW2And1099IntStateOptions`
              ),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-pfd-income`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-state-income-scope-ko`);
          }
        });

        it(`goes from APF to confirmation if spouse has APF and filers are from Alaska`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
              [`/spouseReceivedAlaskaPfd`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-pfd-income`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-you-state-scope-confirm`);
          }
        });

        it(`goes from APF to confirmation if spouse has no APF and filers are from Alaska`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
              [`/spouseReceivedAlaskaPfd`]: createBooleanWrapper(false),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-pfd-income`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-you-state-scope-confirm`);
          }
        });

        it(`goes from APF to knockout if spouse has APF but filers are not from Alaska`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
              [`/spouseReceivedAlaskaPfd`]: createBooleanWrapper(true),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-pfd-income`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-state-income-scope-ko`);
          }
        });

        it(`moves to spouse tax id after the mfj basic info`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
            [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`1958-03-20`),
            [`/secondaryFilerDateOfDeath`]: createDayWrapper(`2022-03-20`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-tax-id`);
        });

        it(`knocks out the filer if the spouse is younger than 16 and the filer wants to file jointly`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
            [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
            [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`2020-03-20`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-age-ko`);
        });

        it(
          `moves to spouse tax id if the filer is widowed and` +
            `skips the spouse age at death if the spouse was older than 65 when they died in the tax year`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
              [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`1950-03-20`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-tax-id`);
          }
        );

        it(
          `moves to spouse tax id if the filer is widowed and` +
            `skips the spouse age at death if the spouse was younger than 65 when they died in the tax year`,
          ({ task }) => {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/maritalStatus`]: createEnumWrapper(`widowed`, `/maritalStatusOptions`),
              [`/yearOfSpouseDeath`]: createEnumWrapper(`taxYear`, `/yearOfSpouseDeathOptions`),
              [`/filers/#${spouseId}/dateOfBirth`]: createDayWrapper(`2000-03-20`),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-tax-id`);
          }
        );

        it(`moves to TIN from basic info in MFJ`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-basic-info`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-tax-id`);
          }
        });

        it(`moves to SSN valid for work from tax id if the filer is not a citizen all year`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-tax-id`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`);
          }
        });

        it(`moves to SSN federal benefits from valid for work if the spouse is not valid for employment`, ({
          task,
        }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
              [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(`notValid`, `/ssnEmploymentValidityOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ssn-federal-benefits`);
          }
        });

        it(`moves to IP pin choice from valid for work if the filer is anything besides not valid for work`, ({
          task,
        }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
              [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(`neither`, `/ssnEmploymentValidityOptions`),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`);

            factGraph.set(
              Path.concretePath(`/secondaryFilerSsnEmploymentValidity`, spouseId),
              EnumFactory(`validOnlyWithDhsAuthorization`, `/ssnEmploymentValidityOptions`).right
            );
            factGraph.save();

            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`);

            factGraph.set(
              Path.concretePath(`/secondaryFilerSsnEmploymentValidity`, spouseId),
              EnumFactory(`validOnlyWithDhsAuthorization`, `/ssnEmploymentValidityOptions`).right
            );
            factGraph.save();

            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`);
          }
        });

        it(`moves to the IP PIN choice question from ssn federal benefits`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
              [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
              [`/secondaryFilerSsnEmploymentValidity`]: createEnumWrapper(`notValid`, `/ssnEmploymentValidityOptions`),
              [`/filers/#${spouseId}/writableHasSSNOnlyForBenefits`]: createBooleanWrapper(false),
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-federal-benefits`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`);
          }
        });

        it(`moves to the IP PIN choice question from the TIN for MFJ`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(
                `/flow/you-and-your-family/spouse/spouse-mfj-ssn-valid-for-work`,
                spouseId,
                task
              )
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`);
          }
        });

        it(`moves to the IP PIN ready question if the spouse does have a IP PIN`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready`);
          }
        });

        it(`moves to the IP PIN input if the filer is ready to put in the spouse's IP PIN`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(true),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-input`);
          }
        });

        it(`moves to the IP PIN not ready if the filer is not ready to put in the spouse's IP PIN`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,
              ...twoFilers,
              [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
              [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-not-ready`);
          }
        });

        describe(`moves to the spouse blind question`, () => {
          const expectedNextScreen = `/flow/you-and-your-family/spouse/spouse-mfj-blind`;

          it(`if the spouse does not have a IP PIN`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(`if the filer has entered the spouse's IP PIN`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
                [`/filers/#${spouseId}/identityPin`]: createIpPinWrapper(`123456`),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-input`, spouseId, task)
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(`if the filer moves from the IP PIN not ready`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(true),
                [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-not-ready`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(expectedNextScreen);
            }
          });

          it(
            `even if the filer has said they are ready to fill in the spouse's IP PIN` +
              ` and then went back and said the spouse doesn't have an IP PIN`,
            ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/hasIpPin`]: createBooleanWrapper(false),
                  [`/filers/#${spouseId}/flowSpouseIpPinReady`]: createBooleanWrapper(true),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(
                    `/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-choice`,
                    spouseId,
                    task
                  )
                ).toRouteNextTo(expectedNextScreen);
              }
            }
          );
        });

        it(`moves to the spouse self-care question regardless of the answer to blindness`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,

              ...twoFilers,
              [`/filers/#${spouseId}/isBlind`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-blind`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-self-care`);

            factGraph.set(Path.concretePath(`/filers/*/isBlind`, spouseId), true);
            factGraph.save();
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-blind`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-self-care`);
          }
        });

        it(`moves to the student question regardless of the answer to self-care`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,

              ...twoFilers,
              [`/filers/#${spouseId}/isBlind`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-self-care`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ft-student`);

            factGraph.set(Path.concretePath(`/filers/*/isDisabled`, spouseId), true);
            factGraph.save();
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-self-care`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-ft-student`);
          }
        });

        it(`moves to the claim question regardless of the answer to disability`, ({ task }) => {
          for (const maritalStatus of maritalStatusVariations) {
            const { factGraph } = setupFactGraph({
              ...modifiedBaseFilerData,

              ...twoFilers,
              [`/filers/#${spouseId}/isBlind`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isDisabled`]: createBooleanWrapper(false),
              [`/filers/#${spouseId}/isStudent`]: createBooleanWrapper(false),
              ...maritalStatus.facts,
            });
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ft-student`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-could-be-claimed`);

            factGraph.set(Path.concretePath(`/filers/*/isStudent`, spouseId), true);
            factGraph.save();
            expect(
              givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-ft-student`, spouseId, task)
            ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-could-be-claimed`);
          }
        });

        describe(`The "claim" section of for a spouse in MFJ`, () => {
          const baseMfjdFiler = {
            ...modifiedBaseFilerData,
            ...twoFilers,
            [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
            [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
            [`/MFJRequiredToFile`]: createBooleanWrapper(false),
            [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
            [`/MFJDepedentsEnrolledMarketplacePlan`]: createBooleanWrapper(true),
            [`/advancedPTCPaymentsMade`]: createBooleanWrapper(true),
          };

          it(`skips all claim questions is canBeClaimed is "no"`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-could-be-claimed`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
            }
          });

          it(`moves to spouse-mfj-spouse-filing-requirement if spouse-mfj-could-be-claimed is "yes"`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });

              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-could-be-claimed`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`);
            }
          });

          it(
            `moves to dependent taxpayer intro if spouse-mfj-spouse-filing-requirement` +
              ` is "no" and either filer can be claimed as dependents or need to file a tax return`,
            ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
                  [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  ...maritalStatus.facts,
                });

                expect(
                  givenFacts(factGraph).atPath(
                    `/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`,
                    spouseId,
                    task
                  )
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-dep-tp-intro`);

                factGraph.set(Path.concretePath(`/filers/*/canBeClaimed`, primaryFilerId), false);
                factGraph.set(Path.concretePath(`/filers/*/canBeClaimed`, spouseId), true);
                factGraph.save();

                expect(
                  givenFacts(factGraph).atPath(
                    `/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`,
                    spouseId,
                    task
                  )
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-dep-tp-intro`);
              }
            }
          );

          it(`moves to refund only from the dependent taxpayer intro`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
                [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });

              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-dep-tp-intro`);
            }
          });

          it(
            `moves to spouse dataview if spouse-mfj-spouse-filing-requirement is "yes",` +
              ` even if the primary filer can be claimed and must file and the secondary filer can not be claimed`,
            ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(true),
                  ...maritalStatus.facts,
                });

                expect(
                  givenFacts(factGraph).atPath(
                    `/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`,
                    spouseId,
                    task
                  )
                ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
              }
            }
          );

          it(`skips all claim questions is spouse-mfj-spouse-filing-requirement is "yes"`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-spouse-filing-requirement`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
            }
          });

          it(`moves to mfjd-breather if TP doesn't want to claim tax benefits and secondary cannot be claimed,`, ({
            task,
          }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/filers/#${primaryFilerId}/potentialClaimerMustFile`]: createBooleanWrapper(true),
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(false),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-refund-only`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-breather`);
            }
          });

          it(`moves to mfj-will-be-claimed if TP doesn't want to claim tax benefits and secondary can be claimed,`, ({
            task,
          }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-refund-only`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-mfj-will-be-claimed`);
            }
          });

          it(`moves to mfjd-breather after mfj-will-be-claimed,`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(
                  `/flow/you-and-your-family/spouse/spouse-mfj-will-be-claimed`,
                  spouseId,
                  task
                )
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-breather`);
            }
          });

          describe(`refusal to let another TP claim primary or seconadary as dependents`, () => {
            it(`moves to mfjd-marketplace-plan`, ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-breather`, spouseId, task)
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-marketplace-plan`);
              }
            });

            it(`moves to mfjd-aptc-paid if either primary or secondary enrolled themselves or anyone else in
              a qualified marketplace plan, yes`, ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                  [`/MFJDepedentsEnrolledMarketplacePlan`]: createBooleanWrapper(true),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-marketplace-plan`, spouseId, task)
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-aptc-paid`);
              }
            });

            it(`moves to mfjd-ok if either primary or secondary enrolled themselves or anyone else in a
              qualified marketplace plan, no`, ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                  [`/MFJDepedentsEnrolledMarketplacePlan`]: createBooleanWrapper(false),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-marketplace-plan`, spouseId, task)
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-ok-1`);
              }
            });

            it(`moves to mfjd-enrollees if Advanced PTC payments made, yes`, ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                  [`/MFJDepedentsEnrolledMarketplacePlan`]: createBooleanWrapper(true),
                  [`/advancedPTCPaymentsMade`]: createBooleanWrapper(true),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-aptc-paid`, spouseId, task)
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-enrollees`);
              }
            });

            describe(`when enrollees don't contain other`, () => {
              it(`moves to mfjd-switch-not-dependent-tp`, ({ task }) => {
                const scenarios = [
                  {
                    enrollees: [`primary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                  },
                  {
                    enrollees: [`primary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                  },
                  {
                    enrollees: [`secondary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                  },
                  {
                    enrollees: [`secondary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                  },
                ];

                for (const maritalStatus of maritalStatusVariations) {
                  scenarios.forEach((scenario, i) => {
                    const { factGraph } = setupFactGraph({
                      ...baseMfjdFiler,
                      ...maritalStatus.facts,
                      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isPrimaryClaimed),
                      [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(scenario.isSecondaryClaimed),
                      [`/mfjdEnrollees`]: createMultEnumWrapper(scenario.enrollees, `/mfjdEnrolleesOptions`),
                    });
                    expect(
                      givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-enrollees`, spouseId, task)
                    ).toRouteNextTo(
                      `/flow/you-and-your-family/spouse/mfjd-switch-not-dependent-tp`,
                      `Error in test scenario at index ${i}`
                    );
                  });
                }
              });

              it(`moves to mfjd-ok-form-not-required`, ({ task }) => {
                const scenarios = [
                  {
                    enrollees: [`primary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                  },
                  {
                    enrollees: [`primary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                  },
                  {
                    enrollees: [`secondary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                  },
                  {
                    enrollees: [`secondary`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                  },
                ];

                for (const maritalStatus of maritalStatusVariations) {
                  scenarios.forEach((scenario, i) => {
                    const { factGraph } = setupFactGraph({
                      ...baseMfjdFiler,
                      ...maritalStatus.facts,
                      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isPrimaryClaimed),
                      [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(scenario.isSecondaryClaimed),
                      [`/mfjdEnrollees`]: createMultEnumWrapper(scenario.enrollees, `/mfjdEnrolleesOptions`),
                    });
                    expect(
                      givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-enrollees`, spouseId, task)
                    ).toRouteNextTo(
                      `/flow/you-and-your-family/spouse/mfjd-ok-form-not-required`,
                      `Error in test scenario at index ${i}`
                    );
                  });
                }
              });
            });

            describe(`when enrollees contain others`, () => {
              it(`moves to mfjd-ok-form-not-required`, ({ task }) => {
                const scenarios = [
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                ];

                for (const maritalStatus of maritalStatusVariations) {
                  scenarios.forEach((scenario, i) => {
                    const { factGraph } = setupFactGraph({
                      ...baseMfjdFiler,
                      ...maritalStatus.facts,
                      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isPrimaryClaimed),
                      [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(scenario.isSecondaryClaimed),
                      [`/mfjdEnrollees`]: createMultEnumWrapper(scenario.enrollees, `/mfjdEnrolleesOptions`),
                      [`/mfjdOtherTaxFamily`]: createBooleanWrapper(scenario.partOfDifferentTaxFamily),
                    });
                    expect(
                      givenFacts(factGraph).atPath(
                        `/flow/you-and-your-family/spouse/mfjd-other-tax-family`,
                        spouseId,
                        task
                      )
                    ).toRouteNextTo(
                      `/flow/you-and-your-family/spouse/mfjd-ok-form-not-required`,
                      `Error in test scenario at index ${i}`
                    );
                  });
                }
              });

              it(`moves to mfjd-switch-not-dependent-tp`, ({ task }) => {
                const scenarios = [
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: false,
                  },
                  {
                    enrollees: [`primary`, `secondary`, `someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: false,
                  },
                ];

                for (const maritalStatus of maritalStatusVariations) {
                  scenarios.forEach((scenario, i) => {
                    const { factGraph } = setupFactGraph({
                      ...baseMfjdFiler,
                      ...maritalStatus.facts,
                      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isPrimaryClaimed),
                      [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(scenario.isSecondaryClaimed),
                      [`/mfjdEnrollees`]: createMultEnumWrapper(scenario.enrollees, `/mfjdEnrolleesOptions`),
                      [`/mfjdOtherTaxFamily`]: createBooleanWrapper(scenario.partOfDifferentTaxFamily),
                    });
                    expect(
                      givenFacts(factGraph).atPath(
                        `/flow/you-and-your-family/spouse/mfjd-other-tax-family`,
                        spouseId,
                        task
                      )
                    ).toRouteNextTo(
                      `/flow/you-and-your-family/spouse/mfjd-switch-not-dependent-tp`,
                      `Error in test scenario at index ${i}`
                    );
                  });
                }
              });

              it(`moves to mfjd-ok-2`, ({ task }) => {
                const scenarios = [
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: true,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: true,
                    partOfDifferentTaxFamily: true,
                  },
                  {
                    enrollees: [`someoneElseThatPrimaryOrSecondaryEnrolled`],
                    isPrimaryClaimed: false,
                    isSecondaryClaimed: false,
                    partOfDifferentTaxFamily: true,
                  },
                ];

                for (const maritalStatus of maritalStatusVariations) {
                  scenarios.forEach((scenario, i) => {
                    const { factGraph } = setupFactGraph({
                      ...baseMfjdFiler,
                      ...maritalStatus.facts,
                      [`/filers/#${primaryFilerId}/willBeClaimed`]: createBooleanWrapper(scenario.isPrimaryClaimed),
                      [`/filers/#${spouseId}/willBeClaimed`]: createBooleanWrapper(scenario.isSecondaryClaimed),
                      [`/mfjdEnrollees`]: createMultEnumWrapper(scenario.enrollees, `/mfjdEnrolleesOptions`),
                      [`/mfjdOtherTaxFamily`]: createBooleanWrapper(scenario.partOfDifferentTaxFamily),
                    });
                    expect(
                      givenFacts(factGraph).atPath(
                        `/flow/you-and-your-family/spouse/mfjd-other-tax-family`,
                        spouseId,
                        task
                      )
                    ).toRouteNextTo(
                      `/flow/you-and-your-family/spouse/mfjd-ok-2`,
                      `Error in test scenario at index ${i}`
                    );
                  });
                }
              });
            });

            it(`moves to mfjd-ok if Advanced PTC payments made, no`, ({ task }) => {
              for (const maritalStatus of maritalStatusVariations) {
                const { factGraph } = setupFactGraph({
                  ...modifiedBaseFilerData,
                  ...twoFilers,
                  [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                  [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                  [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                  [`/MFJDepedentsEnrolledMarketplacePlan`]: createBooleanWrapper(true),
                  [`/advancedPTCPaymentsMade`]: createBooleanWrapper(false),
                  ...maritalStatus.facts,
                });
                expect(
                  givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfjd-aptc-paid`, spouseId, task)
                ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfjd-ok-1`);
              }
            });

            // Todo: Add the rest of the screens of PTC Section 1
          });

          it(`moves to mfj-dependent-choice-b if potential claimer filed only for refund is no,`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,
                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });
              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-mfj-refund-only`, spouseId, task)
              ).toRouteNextTo(`/flow/you-and-your-family/spouse/mfj-dependent-choice-b`);
            }
          });

          it(`continues to the outro from mfj-dependent-choice-a`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,

                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(false),
                ...maritalStatus.facts,
              });

              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfj-dependent-choice-a`, spouseId, task)
              ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
            }
          });

          it(`continues to the outro from mfj-dependent-choice-b`, ({ task }) => {
            for (const maritalStatus of maritalStatusVariations) {
              const { factGraph } = setupFactGraph({
                ...modifiedBaseFilerData,

                ...twoFilers,
                [`/filers/#${spouseId}/canBeClaimed`]: createBooleanWrapper(true),
                [`/MFJRequiredToFile`]: createBooleanWrapper(false),
                [`/MFJDependentsFilingForCredits`]: createBooleanWrapper(true),
                ...maritalStatus.facts,
              });

              expect(
                givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/mfj-dependent-choice-b`, spouseId, task)
              ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
            }
          });
        });
      });
    });

    describe(`Citizen flow`, () => {
      const baseSetupWithAdditions = {
        ...modifiedBaseFilerData,
        ...twoFilers,
        [`/filers/#${spouseId}/isUsCitizenFullYear`]: createBooleanWrapper(false),
      };

      it(`moves to citizen by end of tax year screen if not citizen`, ({ task }) => {
        for (const maritalStatus of maritalStatusVariations) {
          const { factGraph } = setupFactGraph({
            ...baseSetupWithAdditions,
            ...maritalStatus.facts,
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-citizenship`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-citizen-by-end-ty`);
        }
      });
      it(`from spouse residency screen navigates you to spouse national`, ({ task }) => {
        for (const maritalStatus of maritalStatusVariations) {
          const { factGraph } = setupFactGraph({
            ...baseSetupWithAdditions,
            ...maritalStatus.facts,
            [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
            [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-residency`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/spouse-national`);
        }
      });
      it(`from spouse residency screen navigates you to outro if spouse was not a resident`, ({ task }) => {
        for (const maritalStatus of maritalStatusVariations) {
          const { factGraph } = setupFactGraph({
            ...baseSetupWithAdditions,
            ...maritalStatus.facts,
            [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(false),
            [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(false),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-residency`, spouseId, task)
          ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
        }
      });
      it(`from spouse national screen navigates you to spouse outro scope regardless of answer`, ({ task }) => {
        for (const maritalStatus of maritalStatusVariations) {
          const { factGraph } = setupFactGraph({
            ...baseSetupWithAdditions,
            ...maritalStatus.facts,
            [`/livedTogetherAllYearWithSpouse`]: createBooleanWrapper(true),
            [`/writableSeparationAgreement`]: createBooleanWrapper(false),
            [`/filers/#${spouseId}/writableCitizenAtEndOfTaxYear`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/writableIsNoncitizenResidentFullYear`]: createBooleanWrapper(true),
            [`/filers/#${spouseId}/writableIsNational`]: createBooleanWrapper(false),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-national`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/filing-status-choice-b`);

          factGraph.set(Path.concretePath(`/secondaryFiler/writableIsNational`, spouseId), true);
          factGraph.save();
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-national`, spouseId, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/filing-status-choice-b`);
        }
      });
    });
  });
});

describe(`The single path`, () => {
  describe(`asks if the filer was in a registered domestic partnership`, () => {
    stateVariations.forEach((stateString) => {
      it(`and if the filer lived in ${stateString}`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...modifiedBaseFilerData,
          [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
          [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
        });
        expect(
          givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, null, task)
        ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner`);
      });
    });
  });

  describe(`moves to the outro if they were not in a registered domestic partnership`, () => {
    stateVariations.forEach((stateString) => {
      it(`and if the filer lived in ${stateString}`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...modifiedBaseFilerData,
          [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
          [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
          [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(false),
        });
        expect(
          givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, null, task)
        ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
      });
    });
  });

  describe(`knocks out of the filer if they were in a registered domestic partnership`, () => {
    stateVariations.forEach((stateString) => {
      it(`and if the filer lived in ${stateString}`, ({ task }) => {
        const { factGraph } = setupFactGraph({
          ...modifiedBaseFilerData,
          [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
          [`/maritalStatus`]: createEnumWrapper(`single`, `/maritalStatusOptions`),
          [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(true),
        });
        expect(
          givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, null, task)
        ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner-ko`);
      });
    });
  });

  describe(`The divorced or legally separated path`, () => {
    describe(`asks if the filer was in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(`and if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`divorced`, `/maritalStatusOptions`),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, null, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner`);
        });
      });
    });

    describe(`moves to the outro if they were not in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(`and if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`divorced`, `/maritalStatusOptions`),
            [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(false),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, null, task)
          ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
        });
      });
    });

    describe(`knocks out of the filer if they were in a registered domestic partnership`, () => {
      stateVariations.forEach((stateString) => {
        it(`and if the filer lived in ${stateString}`, ({ task }) => {
          const { factGraph } = setupFactGraph({
            ...modifiedBaseFilerData,
            [`/filerResidenceAndIncomeState`]: createEnumWrapper(stateString, `/scopedStateOptions`),
            [`/maritalStatus`]: createEnumWrapper(`divorced`, `/maritalStatusOptions`),
            [`/inRegisteredDomesticPartnership`]: createBooleanWrapper(true),
          });
          expect(
            givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/registered-domestic-partner`, null, task)
          ).toRouteNextTo(`/flow/you-and-your-family/spouse/registered-domestic-partner-ko`);
        });
      });
    });

    it(`moves to the outro`, ({ task }) => {
      const { factGraph } = setupFactGraph({
        ...modifiedBaseFilerData,
        [`/maritalStatus`]: createEnumWrapper(`divorced`, `/maritalStatusOptions`),
      });
      expect(
        givenFacts(factGraph).atPath(`/flow/you-and-your-family/spouse/spouse-marital-status`, spouseId, task)
      ).toRouteNextTo(`/data-view/flow/you-and-your-family/spouse`);
    });
  });
});
