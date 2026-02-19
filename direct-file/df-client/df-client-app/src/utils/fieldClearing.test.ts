import { Path } from '../flow/Path.js';
import flowNodes from '../flow/flow.js';
import { createFlowConfig } from '../flow/flowConfig.js';
import { createCollectionWrapper } from '../test/persistenceWrappers.js';
import {
  CLEARABLE_DEPENDENT_FACTS,
  getAllPathsFor1095A,
  getFieldsToClearOnFilingStatusChange,
} from './fieldClearing.js';
import { setupFactGraphDeprecated } from '../test/setupFactGraph.js';
import { describe, expect, it } from 'vitest';

const uuid1 = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
const uuid2 = `1b1e355e-3d19-415d-8470-fbafd9f58361`;

const { factGraph } = setupFactGraphDeprecated({
  '/familyAndHousehold': createCollectionWrapper([uuid1, uuid2]),
});

describe(`Field Clearing`, () => {
  it(`Reads deductions and credits sections`, () => {
    expect(() => getFieldsToClearOnFilingStatusChange(factGraph)).not.toThrow();
  });
  it(`Will create a concrete path for each dependent`, () => {
    const fields = getFieldsToClearOnFilingStatusChange(factGraph);
    expect(fields).toContain(Path.concretePath(`/familyAndHousehold/*/relationshipCategory`, uuid1));
    expect(fields).toContain(Path.concretePath(`/familyAndHousehold/*/relationshipCategory`, uuid2));
  });
  it(`Credits and deductions contain only expected abstract paths`, () => {
    // we know that we're safe to ignore the tin and ip pin on a dependent without resetting
    // but if someone adds another abstract path into this section, they should ask "should this be cleared"
    // you should only add it to this list if you DO NOT want that fact to be cleared when filing status changes
    const flow = createFlowConfig(flowNodes);
    const deductionsSubcategory = flow.subcategoriesByRoute.get(`/flow/credits-and-deductions/deductions`);
    const creditsSubcategory = flow.subcategoriesByRoute.get(`/flow/credits-and-deductions/credits`);
    if (!deductionsSubcategory || !creditsSubcategory) {
      throw new Error(`Credits and deductions subcategories must exist`);
    }
    const expectedAbstractPaths = [
      `/familyAndHousehold/*/tin`,
      `/familyAndHousehold/*/dateOfDeath`,
      `/familyAndHousehold/*/hasIpPin`,
      `/familyAndHousehold/*/flowIpPinReady`,
      `/familyAndHousehold/*/identityPin`,
      `/filers/*/hasPhysicianStatementBefore1983`,
      `/filers/*/hasPhysicianStatementBothDisabledAndWillNotImprove`,
      `/filers/*/hasSelfCertPhysStatmntOrVetDisbltyVerified`,
      `/filers/*/arePaymentsTaxDisabilityIncome`,
      `/filers/*/employerHasMandatoryRetirementAge`,
      `/filers/*/hasMetEmployerMandatoryRetirementAge`,
      `/filers/*/isRetOnPermOrTotalDisability`,
      `/filers/*/writableTotalTaxableDisabilityAmount`,
      `/cdccCareProviders/*/writableAmountPaidForCare`,
      `/cdccCareProviders/*/writableEin`,
      `/cdccCareProviders/*/writableEmployerWhoFurnishedCare`,
      `/cdccCareProviders/*/isEmployerFurnished`,
      `/cdccCareProviders/*/writableFirstName`,
      `/cdccCareProviders/*/hasTinOrEin`,
      `/cdccCareProviders/*/hasW2Employer`,
      `/cdccCareProviders/*/writableIsOrganization`,
      `/cdccCareProviders/*/writableIsTaxExempt`,
      `/cdccCareProviders/*/writableLastName`,
      `/cdccCareProviders/*/writableTin`,
      `/cdccCareProviders/*/writableAddress`,
      `/cdccCareProviders/*/writableDueDiligence`,
      `/cdccCareProviders/*/writableOrganizationName`,
      `/filers/*/cdccHadExpensesPaidToQualifyingProvider`,
      `/filers/*/cdccHasDependentCareExpenses`,
      `/filers/*/writableCdccQualifyingExpenseAmount`,
    ];
    const deductionFacts = deductionsSubcategory.screens
      .flatMap((sc) => sc.factPaths)
      .filter((p) => Path.isAbstract(p));
    let creditsFacts = creditsSubcategory.screens.flatMap((sc) => sc.factPaths).filter((p) => Path.isAbstract(p));
    const all1095AFacts = getAllPathsFor1095A(flow);
    creditsFacts = creditsFacts.filter((p) => !all1095AFacts.includes(p));

    // use a Set here to account for duplicated conditional fact paths
    const allAbstractPaths = Array.from(new Set([...deductionFacts, ...creditsFacts]));

    const pathsWithClearableFieldsRemoved = allAbstractPaths.filter((p) => !CLEARABLE_DEPENDENT_FACTS.includes(p));
    expect(pathsWithClearableFieldsRemoved.sort()).toEqual(expectedAbstractPaths.sort());
  });
});
