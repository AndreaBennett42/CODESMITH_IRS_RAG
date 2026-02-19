import { CURRENT_TAX_YEAR } from '../../constants/taxConstants.js';
import { Path } from '../../flow/Path.js';
import {
  createStringWrapper,
  createEnumWrapper,
  createBooleanWrapper,
  createCollectionWrapper,
  createDayWrapper,
} from '../persistenceWrappers.js';
import { describe, it, expect } from 'vitest';
import { setupFactGraph } from '../setupFactGraph.js';
const childId = `4fa3a5a7-a9d1-43a9-a0fb-277596e70d48`;
const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;
const spouseId = `859c03d1-af4a-447f-96aa-d19397048a48`;
const primaryFilerId = uuid;

const childBaseData = {
  '/filers': createCollectionWrapper([primaryFilerId, spouseId]),
  [`/filers/#${primaryFilerId}/isPrimaryFiler`]: createBooleanWrapper(true),
  [`/filers/#${primaryFilerId}/canBeClaimed`]: createBooleanWrapper(false),
  '/maritalStatus': createEnumWrapper(`single`, `/maritalStatusOptions`),
  '/filingStatus': createEnumWrapper(`single`, `/filingStatusOptions`),
  [`/filers/#${primaryFilerId}/dateOfBirth`]: createDayWrapper(`0001-01-01`),
  '/familyAndHousehold': createCollectionWrapper([childId]),

  [`/familyAndHousehold/#${childId}/firstName`]: createStringWrapper(`Child`),
  [`/familyAndHousehold/#${childId}/writableMiddleInitial`]: createStringWrapper(`E`),
  [`/familyAndHousehold/#${childId}/lastName`]: createStringWrapper(`ChildFace`),
  [`/familyAndHousehold/#${childId}/deceased`]: createBooleanWrapper(false),
  [`/familyAndHousehold/#${childId}/isUsCitizenFullYear`]: createBooleanWrapper(true),
};

const biologicalChildBaseData = {
  ...childBaseData,
  [`/familyAndHousehold/#${childId}/relationshipCategory`]: createEnumWrapper(
    `childOrDescendants`,
    `/relationshipCategoryOptions`
  ),
  [`/familyAndHousehold/#${childId}/childRelationship`]: createEnumWrapper(
    `biologicalChild`,
    `/childRelationshipOptions`
  ),
};

describe(`Leap year`, () => {
  it(`Tax year is a leap year when the current tax year is divisible by 4`, ({ task }) => {
    task.meta.testedFactPaths = [`/isTaxYearLeapYear`];
    // This will eventually fail in 2032 if we don't update the /isTaxYearLeapYear
    // code to calculate the leap year correctly (either add another case or update it to use the correct calc)
    const { factGraph } = setupFactGraph({});
    if (Number.parseInt(CURRENT_TAX_YEAR) % 4 === 0) {
      expect(factGraph.get(Path.concretePath(`/isTaxYearLeapYear`, null)).get).toBe(true);
    } else {
      expect(factGraph.get(Path.concretePath(`/isTaxYearLeapYear`, null)).get).toBe(false);
    }
  });
});

describe(`Special Benefit split`, () => {
  it(`Meets the entry condition for a person who is totally disabled`, ({ task }) => {
    task.meta.testedFactPaths = [`/familyAndHousehold/*/childMayQualifyForBenefitSplit`];
    const { factGraph } = setupFactGraph({
      ...biologicalChildBaseData,
      [`/familyAndHousehold/#${childId}/dateOfBirth`]: createDayWrapper(
        `${Number.parseInt(CURRENT_TAX_YEAR) - 50}-01-02`
      ),
      [`/familyAndHousehold/#${childId}/permanentTotalDisability`]: createBooleanWrapper(true),
    });
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/childMayQualifyForBenefitSplit`, childId)).get).toBe(
      true
    );
  });

  it(`Always fails the entry condition for a non-disabled person who turns 18 on July 2`, ({ task }) => {
    task.meta.testedFactPaths = [`/familyAndHousehold/*/childMayQualifyForBenefitSplit`];
    const { factGraph } = setupFactGraph({
      ...biologicalChildBaseData,
      [`/familyAndHousehold/#${childId}/dateOfBirth`]: createDayWrapper(
        `${Number.parseInt(CURRENT_TAX_YEAR) - 18}-07-02`
      ),
      [`/familyAndHousehold/#${childId}/permanentTotalDisability`]: createBooleanWrapper(false),
    });
    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/childMayQualifyForBenefitSplit`, childId)).get).toBe(
      false
    );
  });

  it(`Meets the entry condition for a non-disabled person who turns 18 on July 3`, ({ task }) => {
    task.meta.testedFactPaths = [`/familyAndHousehold/*/childMayQualifyForBenefitSplit`];
    const { factGraph } = setupFactGraph({
      ...biologicalChildBaseData,
      [`/familyAndHousehold/#${childId}/dateOfBirth`]: createDayWrapper(
        `${Number.parseInt(CURRENT_TAX_YEAR) - 18}-07-03`
      ),
      [`/familyAndHousehold/#${childId}/permanentTotalDisability`]: createBooleanWrapper(false),
    });

    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/childMayQualifyForBenefitSplit`, childId)).get).toBe(
      true
    );
  });

  it(`Allows disability to not be set on an 18 year old born in the latter half of the year`, ({ task }) => {
    task.meta.testedFactPaths = [`/familyAndHousehold/*/childMayQualifyForBenefitSplit`];
    const { factGraph } = setupFactGraph({
      ...biologicalChildBaseData,
      [`/familyAndHousehold/#${childId}/dateOfBirth`]: createDayWrapper(
        `${Number.parseInt(CURRENT_TAX_YEAR) - 18}-07-03`
      ),
    });

    expect(factGraph.get(Path.concretePath(`/familyAndHousehold/*/childMayQualifyForBenefitSplit`, childId)).get).toBe(
      true
    );
  });
});
