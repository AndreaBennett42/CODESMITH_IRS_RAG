import { it, describe, expect } from 'vitest';
import * as sfg from '@irs/js-factgraph-scala';
import { baseFilerData, primaryFilerId } from '../testData.js';
import {
  createBooleanWrapper,
  createCollectionWrapper,
  createDollarWrapper,
  createEnumWrapper,
  createTinWrapper,
} from '../persistenceWrappers.js';
import { Path } from '../../flow/Path.js';
import { setupFactGraph } from '../setupFactGraph.js';

const w2Id = `93a9f60d-e491-4e23-b02d-ade63dce5886`;

describe(`W2 TIN`, () => {
  it(`W2 is SSN and is updated`, ({ task }) => {
    task.meta.testedFactPaths = [`formW2s/*/usedTin`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/formW2s': createCollectionWrapper([w2Id]),
      [`/formW2s/#${w2Id}/filer`]: {
        item: {
          id: primaryFilerId,
        },
        $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
      },
    });

    const initialSsn = factGraph.get(Path.concretePath(`/formW2s/*/usedTin`, w2Id));
    expect(initialSsn.complete).toBe(true);
    const initialSsnValue = initialSsn.get;
    expect(initialSsnValue.area).toBe(`555`);
    expect(initialSsnValue.group).toBe(`55`);
    expect(initialSsnValue.serial).toBe(`5555`);

    factGraph.set(Path.concretePath(`/filers/*/tin`, primaryFilerId), sfg.TinFactory(`777777777`).right);
    factGraph.save();
    const updatedSsn = factGraph.get(Path.concretePath(`/formW2s/*/usedTin`, w2Id));
    expect(updatedSsn.complete).toBe(true);
    const updatedSsnValue = updatedSsn.get;
    expect(updatedSsnValue.area).toBe(`777`);
    expect(updatedSsnValue.group).toBe(`77`);
    expect(updatedSsnValue.serial).toBe(`7777`);
  });

  it(`W2 is ITIN and is updated to SSN`, ({ task }) => {
    task.meta.testedFactPaths = [`formW2s/*/filer/usedTin`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      [`/filers/#${primaryFilerId}/tin`]: createTinWrapper({ area: `955`, group: `55`, serial: `5555` }),
      '/formW2s': createCollectionWrapper([w2Id]),
      [`/formW2s/#${w2Id}/filer`]: {
        item: {
          id: primaryFilerId,
        },
        $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
      },
    });

    const initialSsn = factGraph.get(Path.concretePath(`/formW2s/*/usedTin`, w2Id));
    expect(initialSsn.complete).toBe(false);

    factGraph.set(Path.concretePath(`/formW2s/*/tin`, w2Id), sfg.TinFactory(`777777777`).right);
    factGraph.save();
    const tinInfo = factGraph.get(Path.concretePath(`/formW2s/*/usedTin`, w2Id));
    expect(tinInfo.complete).toBe(true);
    const tinInfoValue = tinInfo.get;
    expect(tinInfoValue.area).toBe(`777`);
    expect(tinInfoValue.group).toBe(`77`);
    expect(tinInfoValue.serial).toBe(`7777`);

    // Updates TIN in about you section
    factGraph.set(Path.concretePath(`/filers/*/tin`, primaryFilerId), sfg.TinFactory(`111111111`).right);
    factGraph.save();
    const updatedSsn = factGraph.get(Path.concretePath(`/formW2s/*/usedTin`, w2Id));
    expect(updatedSsn.complete).toBe(true);
    const updatedSsnValue = updatedSsn.get;
    expect(updatedSsnValue.area).toBe(`111`);
    expect(updatedSsnValue.group).toBe(`11`);
    expect(updatedSsnValue.serial).toBe(`1111`);
  });
});

describe(`W-2 validations`, () => {
  describe(`/formW2s/*/hasStateWagesWithoutStateCode`, () => {
    it.each([
      [undefined, undefined, false, false],
      [undefined, `1000.00`, true, true],
      [`sameState`, undefined, false, false],
      [`sameState`, `1000.00`, false, false],
      [`differentState`, undefined, false, false],
      [`differentState`, `1000.00`, false, false],
    ])(
      `When state is %s, and state wages is %s, then /hasStateWagesWithoutStateCode should be %s`,
      (stateValue, stateWages, hasStateWagesWithoutStateCodeExpectedValue, hasW2ValidationErrorsExpectedValue) => {
        const { factGraph } = setupFactGraph({
          ...baseFilerData,
          '/formW2s': createCollectionWrapper([w2Id]),
          [`/formW2s/#${w2Id}/writableState`]: stateValue
            ? createEnumWrapper(stateValue, `/incomeFormStateOptions`)
            : undefined,
          [`/formW2s/#${w2Id}/writableStateWages`]: stateWages ? createDollarWrapper(stateWages) : undefined,
        });

        const hasStateWagesWithoutStateCode = factGraph.get(
          Path.concretePath(`/formW2s/*/hasStateWagesWithoutStateCode`, w2Id)
        );
        const hasW2ValidationErrors = factGraph.get(Path.concretePath(`/hasW2ValidationErrors`, null));

        expect(hasStateWagesWithoutStateCode.get).toEqual(hasStateWagesWithoutStateCodeExpectedValue);
        expect(hasW2ValidationErrors.get).toEqual(hasW2ValidationErrorsExpectedValue);
      }
    );
  });

  describe(`/formW2s/*/box17GreaterThanBox16`, () => {
    it.each([
      [undefined, undefined, false, false],
      [`1000.00`, undefined, false, false],
      [undefined, `1000.00`, false, false],
      [`1000.00`, `1000.00`, false, false],
      [`1000.00`, `999.00`, false, false],
      [`1000.00`, `1001.00`, true, true],
    ])(
      `When box 16 is %s, and box 17 is %s, then /box17GreaterThanBox16 should be %s`,
      (box16Value, box17Value, box17GreaterThanBox16ExpectedValue, hasW2ValidationErrorsExpectedValue) => {
        const { factGraph } = setupFactGraph({
          ...baseFilerData,
          '/formW2s': createCollectionWrapper([w2Id]),
          [`/formW2s/#${w2Id}/writableState`]: createEnumWrapper(`sameState`, `/incomeFormStateOptions`),
          [`/formW2s/#${w2Id}/writableStateWages`]: box16Value ? createDollarWrapper(box16Value) : undefined,
          [`/formW2s/#${w2Id}/writableStateWithholding`]: box17Value ? createDollarWrapper(box17Value) : undefined,
        });

        const box17GreaterThanBox16 = factGraph.get(Path.concretePath(`/formW2s/*/box17GreaterThanBox16`, w2Id));
        const hasW2ValidationErrors = factGraph.get(Path.concretePath(`/hasW2ValidationErrors`, null));

        expect(box17GreaterThanBox16.get).toEqual(box17GreaterThanBox16ExpectedValue);
        expect(hasW2ValidationErrors.get).toEqual(hasW2ValidationErrorsExpectedValue);
      }
    );
  });

  describe(`/formW2s/*/box19GreaterThanBox18`, () => {
    it.each([
      [undefined, undefined, false, false],
      [`1000.00`, undefined, false, false],
      [undefined, `1000.00`, false, false],
      [`1000.00`, `1000.00`, false, false],
      [`1000.00`, `999.00`, false, false],
      [`1000.00`, `1001.00`, true, true],
    ])(
      `When box 19 is %s, and box 18 is %s, then /box19GreaterThanBox18 should be %s`,
      (box18Value, box19Value, box19GreaterThanBox18ExpectedValue, hasW2ValidationErrorsExpectedValue) => {
        const { factGraph } = setupFactGraph({
          ...baseFilerData,
          '/formW2s': createCollectionWrapper([w2Id]),
          [`/formW2s/#${w2Id}/writableLocalWages`]: box18Value ? createDollarWrapper(box18Value) : undefined,
          [`/formW2s/#${w2Id}/writableLocalWithholding`]: box19Value ? createDollarWrapper(box19Value) : undefined,
        });

        const box19GreaterThanBox18 = factGraph.get(Path.concretePath(`/formW2s/*/box19GreaterThanBox18`, w2Id));
        const hasW2ValidationErrors = factGraph.get(Path.concretePath(`/hasW2ValidationErrors`, null));

        expect(box19GreaterThanBox18.get).toEqual(box19GreaterThanBox18ExpectedValue);
        expect(hasW2ValidationErrors.get).toEqual(hasW2ValidationErrorsExpectedValue);
      }
    );
  });
});

describe(`Deferrals and contributions`, () => {
  it(`Includes Code G if the plan is governmental`, ({ task }) => {
    task.meta.testedFactPaths = [`/formW2s/*/totalDeferralsAndContributions`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/formW2s': createCollectionWrapper([w2Id]),
      [`/formW2s/#${w2Id}/401kDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/403bDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/sarsepDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/457bDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/457bIsGovernmentalPlan`]: createBooleanWrapper(true),
      [`/formW2s/#${w2Id}/501c18Deferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/simpleContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth401kContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth403bContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth457bContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/writableCombatPay`]: createDollarWrapper(`1`), // canary value, shouldn't be included
    });
    expect(factGraph.get(Path.concretePath(`/formW2s/*/totalDeferralsAndContributions`, w2Id)).get.toString()).toBe(
      `9000.00`
    );
  });

  it(`Excludes Code G if the plan is non-governmental`, ({ task }) => {
    task.meta.testedFactPaths = [`/formW2s/*/totalDeferralsAndContributions`];
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/formW2s': createCollectionWrapper([w2Id]),
      [`/formW2s/#${w2Id}/401kDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/403bDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/sarsepDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/457bDeferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/457bIsGovernmentalPlan`]: createBooleanWrapper(false),
      [`/formW2s/#${w2Id}/501c18Deferrals`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/simpleContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth401kContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth403bContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/roth457bContributions`]: createDollarWrapper(`1000`),
      [`/formW2s/#${w2Id}/writableCombatPay`]: createDollarWrapper(`1`), // canary value, shouldn't be included
    });
    expect(factGraph.get(Path.concretePath(`/formW2s/*/totalDeferralsAndContributions`, w2Id)).get.toString()).toBe(
      `8000.00`
    );
  });
});
