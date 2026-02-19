import { Path } from '../../flow/Path.js';
import {
  createBooleanWrapper,
  createCollectionWrapper,
  createDollarWrapper,
  createEnumWrapper,
} from '../persistenceWrappers.js';
import { setupFactGraphDeprecated } from '../setupFactGraph.js';
import { baseFilerData, primaryFilerId, marriedFilerData, spouseId, filerWithZeroBalanceData } from '../testData.js';
import { describe, it, expect } from 'vitest';

describe(`Filers have added federal withholding values from 1099Miscs`, () => {
  it(`form1099MiscFederalIncomeTaxWithheld is correct when filer has entered a witholding`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099MiscFederalIncomeTaxWithheld`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Miscs/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`20.22`),
    });

    const form1099MiscFederalIncomeTaxWithheld = factGraph.get(
      Path.concretePath(`/form1099MiscFederalIncomeTaxWithheld`, primaryFilerId)
    );
    expect(form1099MiscFederalIncomeTaxWithheld.get.toString()).toBe(`20.00`);
  });

  it(`form1099MiscFederalIncomeTaxWithheld is correct when filer and spouse have entered a witholding`, ({ task }) => {
    task.meta.testedFactPaths = [`/form1099MiscFederalIncomeTaxWithheld`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      ...marriedFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Miscs/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`20.22`),
      [`/form1099Miscs/#${spouseId}/writableFederalWithholding`]: createDollarWrapper(`40`),
    });

    const form1099MiscFederalIncomeTaxWithheld = factGraph.get(
      Path.concretePath(`/form1099MiscFederalIncomeTaxWithheld`, primaryFilerId)
    );
    expect(form1099MiscFederalIncomeTaxWithheld.get.toString()).toBe(`60.00`);
  });
});

describe(`Filers have APF income`, () => {
  // TODO: This will need to be updated when more 1099Misc fields are added
  // Currently always evaluates to true
  it(`1099Misc isAlaskaPfd`, ({ task }) => {
    task.meta.testedFactPaths = [`form1099Miscs/*/isAlaskaPfd`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
    });

    const isAlaskaPfd = factGraph.get(Path.concretePath(`/form1099Miscs/*/isAlaskaPfd`, primaryFilerId));
    expect(isAlaskaPfd.get).toBe(true);
  });

  it(`The filer has alaskaPfdIncome`, ({ task }) => {
    task.meta.testedFactPaths = [`/alaskaPfdIncome`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
    });

    const alaskaPfdIncome = factGraph.get(Path.concretePath(`/alaskaPfdIncome`, primaryFilerId)).get;
    expect(alaskaPfdIncome.toString()).toBe(`300.00`);
  });

  // TODO: currently isAlaskaPfd always evaluates to true.
  // This will change when the capability exists to add more 1099Miscs
  // it(`The filer does not have alaskaPfdIncome`, ({ task }) => {
  //   task.meta.testedFactPaths = [`/alaskaPfdIncome`];
  //   const { factGraph } = setupFactGraphDeprecated({
  //     ...baseFilerData,
  //     [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
  //     [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(false),
  //   });

  //   const alaskaPfdIncome = factGraph.get(Path.concretePath(`/alaskaPfdIncome`, primaryFilerId)).get;
  //   expect(alaskaPfdIncome.toString()).toBe(`0`);
  // });

  it(`The filer and spouse have alaskaPfdIncome`, ({ task }) => {
    task.meta.testedFactPaths = [`/alaskaPfdIncome`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      ...marriedFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${spouseId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${spouseId}/writableOtherIncome`]: createDollarWrapper(`500`),
    });

    const alaskaPfdIncome = factGraph.get(Path.concretePath(`/alaskaPfdIncome`, primaryFilerId)).get;
    expect(alaskaPfdIncome.toString()).toBe(`800.00`);
  });
});

describe(`Filers have 1099Miscs`, () => {
  it(`The filer has at least one 1099Misc`, ({ task }) => {
    task.meta.testedFactPaths = [`/has1099Misc`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
    });

    const has1099Misc = factGraph.get(Path.concretePath(`/has1099Misc`, primaryFilerId)).get;
    expect(has1099Misc).toBe(true);
  });

  it(`The filer and spouse each have at least one 1099Misc`, ({ task }) => {
    task.meta.testedFactPaths = [`/has1099Misc`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      ...marriedFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
    });

    const has1099MiscPrimary = factGraph.get(Path.concretePath(`/has1099Misc`, primaryFilerId)).get;
    const has1099MiscSpouse = factGraph.get(Path.concretePath(`/has1099Misc`, spouseId)).get;

    expect(has1099MiscPrimary).toBe(true);
    expect(has1099MiscSpouse).toBe(true);
  });

  it(`The filer has no 1099Miscs`, ({ task }) => {
    task.meta.testedFactPaths = [`/has1099Misc`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/form1099Miscs`]: createCollectionWrapper([]),
    });

    const has1099Misc = factGraph.get(Path.concretePath(`/has1099Misc`, primaryFilerId)).get;
    expect(has1099Misc).toBe(false);
  });
});

describe(`APF eligibility`, () => {
  it(`eligibleForApf is true if the TP is from Alaska`, ({ task }) => {
    task.meta.testedFactPaths = [`/eligibleForApf`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
    });

    const eligibleForApf = factGraph.get(Path.concretePath(`/eligibleForApf`, primaryFilerId));
    expect(eligibleForApf.get).toBe(true);
  });

  it(`eligibleForApf is false if the TP is from another scoped state`, ({ task }) => {
    task.meta.testedFactPaths = [`/eligibleForApf`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ca`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
    });

    const eligibleForApf = factGraph.get(Path.concretePath(`/eligibleForApf`, primaryFilerId));
    expect(eligibleForApf.get).toBe(false);
  });
});

describe(`APF section completeness`, () => {
  it(`isApfSectionComplete is complete if TP is not eligible for APF`, ({ task }) => {
    task.meta.testedFactPaths = [`/isApfSectionComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ar`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
    });

    const isApfSectionComplete = factGraph.get(Path.concretePath(`/isApfSectionComplete`, primaryFilerId));
    expect(isApfSectionComplete.get).toBe(true);
  });

  it(`isApfSectionComplete is complete if TP is eligible for APF`, ({ task }) => {
    task.meta.testedFactPaths = [`/isApfSectionComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/hasCompletedApfSection`]: createBooleanWrapper(true),
    });

    const isApfSectionComplete = factGraph.get(Path.concretePath(`/isApfSectionComplete`, primaryFilerId));
    expect(isApfSectionComplete.get).toBe(true);
  });

  it(`isApfSectionComplete is incomplete if TP is eligible for APF and has not filled out APF`, ({ task }) => {
    task.meta.testedFactPaths = [`/isApfSectionComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      ...baseFilerData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
    });

    const isApfSectionComplete = factGraph.get(Path.concretePath(`/isApfSectionComplete`, primaryFilerId));
    expect(isApfSectionComplete.get).toBe(false);
  });
});

describe(`Form1099Misc completeness`, () => {
  it(`APF 1099Misc is complete after seeing last screen and filling out income`, ({ task }) => {
    task.meta.testedFactPaths = [`form1099Miscs/*/isComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${primaryFilerId}/hasSeenLastAvailableScreen`]: createBooleanWrapper(true),
    });

    const isComplete = factGraph.get(Path.concretePath(`/form1099Miscs/*/isComplete`, primaryFilerId));
    expect(isComplete.get).toBe(true);
  });

  it(`APF 1099Misc is incomplete after seeing last screen but not filling out income`, ({ task }) => {
    task.meta.testedFactPaths = [`form1099Miscs/*/isComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/hasSeenLastAvailableScreen`]: createBooleanWrapper(true),
    });

    const isComplete = factGraph.get(Path.concretePath(`/form1099Miscs/*/isComplete`, primaryFilerId));
    expect(isComplete.get).toBe(false);
  });

  it(`APF 1099Misc is incomplete after filling out income but hasn't seen last screen`, ({ task }) => {
    task.meta.testedFactPaths = [`form1099Miscs/*/isComplete`];
    const { factGraph } = setupFactGraphDeprecated({
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
    });

    const isComplete = factGraph.get(Path.concretePath(`/form1099Miscs/*/isComplete`, primaryFilerId));
    expect(isComplete.get).toBe(false);
  });
});

describe(`APF increases total taxable income`, () => {
  it(`APF increases total taxable income for filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/totalIncome`];
    const { factGraph } = setupFactGraphDeprecated({
      ...filerWithZeroBalanceData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
    });

    const totalIncome = factGraph.get(Path.concretePath(`/totalIncome`, primaryFilerId));

    expect(totalIncome.get.toString()).toBe(`200300.00`);
  });

  it(`APF increases total taxable income for filer and spouse`, ({ task }) => {
    task.meta.testedFactPaths = [`/totalIncome`];
    const { factGraph } = setupFactGraphDeprecated({
      ...filerWithZeroBalanceData,
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${spouseId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${spouseId}/writableOtherIncome`]: createDollarWrapper(`300`),
    });

    const totalIncome = factGraph.get(Path.concretePath(`/totalIncome`, primaryFilerId));

    expect(totalIncome.get.toString()).toBe(`200600.00`);
  });
});

describe(`APF withholding increases total withholding`, () => {
  it(`APF increases total withholding for filer`, ({ task }) => {
    task.meta.testedFactPaths = [`/totalWithholding`];
    const { factGraph } = setupFactGraphDeprecated({
      ...filerWithZeroBalanceData,
      [`/formW2s/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`38400`),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`100`),
    });

    const totalWithholding = factGraph.get(Path.concretePath(`/totalWithholding`, primaryFilerId));

    expect(totalWithholding.get.toString()).toBe(`38500.00`);
  });

  it(`APF increases total withholding for filer and spouse`, ({ task }) => {
    task.meta.testedFactPaths = [`/totalWithholding`];
    const { factGraph } = setupFactGraphDeprecated({
      ...filerWithZeroBalanceData,
      [`/formW2s/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`38400`),
      [`/formW2s/#${spouseId}/writableFederalWithholding`]: createDollarWrapper(`38400`),
      [`/filerResidenceAndIncomeState`]: createEnumWrapper(`ak`, `/scopedStateOptions`),
      [`/form1099Miscs`]: createCollectionWrapper([primaryFilerId, spouseId]),
      [`/form1099Miscs/#${primaryFilerId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${primaryFilerId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${primaryFilerId}/writableFederalWithholding`]: createDollarWrapper(`100`),
      [`/form1099Miscs/#${spouseId}/isAlaskaPfd`]: createBooleanWrapper(true),
      [`/form1099Miscs/#${spouseId}/writableOtherIncome`]: createDollarWrapper(`300`),
      [`/form1099Miscs/#${spouseId}/writableFederalWithholding`]: createDollarWrapper(`100`),
    });

    const totalWithholding = factGraph.get(Path.concretePath(`/totalWithholding`, primaryFilerId));

    expect(totalWithholding.get.toString()).toBe(`38600.00`);
  });
});
