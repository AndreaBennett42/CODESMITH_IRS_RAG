import { setupFactGraph } from '../test/setupFactGraph.js';
import { ExportableCollection, getFactKeyFromPath, getStateExportableFactsFromGraph } from './exportUtils.js';
import fs from 'fs';

const SCENARIO_FOLDER = `./src/test/factDictionaryTests/backend-scenarios`;
const readScenarioFile = (fileName: string) => fs.readFileSync(`${SCENARIO_FOLDER}/${fileName}.json`, `utf-8`);

describe(getStateExportableFactsFromGraph.name, () => {
  describe(`Single filer`, () => {
    describe(`no familyAndHousehold`, () => {
      it(`gets the exportable facts and their values from the fact graph`, () => {
        const scenarioFileContents = readScenarioFile(`single-35k`);
        const factJson = JSON.parse(scenarioFileContents);
        const { factGraph } = setupFactGraph(factJson.facts);

        const result = getStateExportableFactsFromGraph(factGraph);

        expect(result).toEqual({
          filers: [
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `SuperLongTwentyChars`,
                sensitive: false,
              },
              middleInitial: {
                value: `E`,
                sensitive: false,
              },
              lastName: {
                value: `SuperLongTwentyChars`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `923-00-6789`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1901-01-01`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `true`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `150.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
          ],
          formW2s: [
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `923-00-6788`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
          ],
          interestReports: [
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `923-00-6789`,
                sensitive: true,
              },
              payer: {
                value: `XYZ Financial`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `150.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `0.00`,
                sensitive: false,
              },
            },
          ],
          socialSecurityReports: [
            {
              formType: {
                sensitive: false,
                value: `SSA-1099`,
              },
              netBenefits: {
                sensitive: false,
                value: `2400.00`,
              },
              recipientTin: {
                sensitive: true,
                value: `923-00-6789`,
              },
            },
          ],
        });
      });
    });

    describe(`with dependent(s)`, () => {
      it(`gets the exportable facts and their values from the fact graph, one dependent`, () => {
        const scenarioFileContents = readScenarioFile(`hoh-35k`);
        const factJson = JSON.parse(scenarioFileContents);
        const { factGraph } = setupFactGraph(factJson.facts);

        const result = getStateExportableFactsFromGraph(factGraph);

        expect(result).toEqual({
          familyAndHousehold: [
            {
              relationship: {
                value: `biologicalChild`,
                sensitive: false,
              },
              firstName: {
                value: `Booper`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Brown`,
                sensitive: false,
              },
              suffix: {
                value: `Jr`,
                sensitive: false,
              },
              tin: {
                value: `222-00-5232`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2021-01-01`,
                sensitive: false,
              },
              residencyDuration: {
                value: `allYear`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `true`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: `true`,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: `twelve`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
          ],
          filers: [
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `Bertha`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Brown`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `333-00-3333`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1980-01-01`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `true`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `500.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
          ],
          formW2s: [
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `333-00-3333`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
          ],
          interestReports: [
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `333-00-3333`,
                sensitive: true,
              },
              payer: {
                value: `Writer Co.`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `500.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `0.00`,
                sensitive: false,
              },
            },
          ],
        });
      });

      it(`gets the exportable facts and their values from the fact graph, multiple familyAndHousehold`, () => {
        const scenarioFileContents = readScenarioFile(`hoh-40k`);
        const factJson = JSON.parse(scenarioFileContents);
        const { factGraph } = setupFactGraph(factJson.facts);

        const result = getStateExportableFactsFromGraph(factGraph);

        expect(result).toEqual({
          familyAndHousehold: [
            {
              relationship: {
                value: `biologicalChild`,
                sensitive: false,
              },
              firstName: {
                value: `Apricot`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Tester`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `123-00-1222`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2003-02-01`,
                sensitive: false,
              },
              residencyDuration: {
                value: `allYear`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `true`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: `true`,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: `twelve`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `true`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
            {
              relationship: {
                value: `biologicalChild`,
                sensitive: false,
              },
              firstName: {
                value: `Apple`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Tester`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `123-00-2011`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2008-01-16`,
                sensitive: false,
              },
              residencyDuration: {
                value: `allYear`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `true`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: `true`,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: `twelve`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
          ],
          filers: [
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `Mango`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Test`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `123-00-1234`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1980-01-01`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `true`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: null,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `1700.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `238.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
          ],
          formW2s: [
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `123-00-1234`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
          ],
          interestReports: [
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `123-00-1234`,
                sensitive: true,
              },
              payer: {
                value: `HSBC`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `137.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `0.00`,
                sensitive: false,
              },
            },
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `123-00-1234`,
                sensitive: true,
              },
              payer: {
                value: `Samsung`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `101.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `0.00`,
                sensitive: false,
              },
            },
          ],
          socialSecurityReports: [
            {
              formType: {
                sensitive: false,
                value: `SSA-1099`,
              },
              netBenefits: {
                sensitive: false,
                value: `377.00`,
              },
              recipientTin: {
                sensitive: true,
                value: `123-00-1234`,
              },
            },
          ],
          form1099Gs: [
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `123-00-1234`,
                sensitive: false,
              },
              payer: {
                value: `yes`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: false,
              },
              amount: {
                value: `1700.00`,
                sensitive: false,
              },
              amountPaidBackForBenefitsInTaxYear: {
                value: `0.00`,
                sensitive: false,
              },
              federalTaxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              stateIdNumber: {
                value: null,
                sensitive: false,
              },
              stateTaxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
            },
          ],
        });
      });
    });
  });

  describe(`Multiple filers`, () => {
    describe(`with dependent(s)`, () => {
      it(`gets the exportable facts and their values from the fact graph, one dependent`, () => {
        const scenarioFileContents = readScenarioFile(`mfj-11k`);
        const factJson = JSON.parse(scenarioFileContents);
        const { factGraph } = setupFactGraph(factJson.facts);

        const result = getStateExportableFactsFromGraph(factGraph);

        expect(result).toEqual({
          familyAndHousehold: [
            {
              relationship: {
                value: `biologicalChild`,
                sensitive: false,
              },
              firstName: {
                value: `Valentina`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Siberia`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `222-00-2222`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2011-08-11`,
                sensitive: false,
              },
              residencyDuration: {
                value: `allYear`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `true`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: null,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: null,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
          ],
          filers: [
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `Boris`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Siberian`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `333-00-3333`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1995-02-10`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `true`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: null,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `30.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `Mickey`,
                sensitive: false,
              },
              middleInitial: {
                value: null,
                sensitive: false,
              },
              lastName: {
                value: `Barry`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `999-00-9999`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1997-06-21`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `false`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
          ],
          formW2s: [
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `333-00-3333`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
          ],
          interestReports: [
            {
              has1099: {
                value: `false`,
                sensitive: false,
              },
              recipientTin: {
                value: `333-00-3333`,
                sensitive: true,
              },
              payer: {
                value: `Purr City Bank`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `30.00`,
                sensitive: false,
              },
            },
          ],
        });
      });

      it(`gets the exportable facts and their values from the fact graph, multiple familyAndHousehold`, () => {
        const scenarioFileContents = readScenarioFile(`mfj-43k`);
        const factJson = JSON.parse(scenarioFileContents);
        const { factGraph } = setupFactGraph(factJson.facts);

        const result = getStateExportableFactsFromGraph(factGraph);

        expect(result).toEqual({
          familyAndHousehold: [
            {
              relationship: {
                value: `biologicalChild`,
                sensitive: false,
              },
              firstName: {
                value: `Lal`,
                sensitive: false,
              },
              middleInitial: {
                value: `J`,
                sensitive: false,
              },
              lastName: {
                value: `Boone`,
                sensitive: false,
              },
              suffix: {
                value: `Jr`,
                sensitive: false,
              },
              tin: {
                value: `102-00-0001`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2016-08-17`,
                sensitive: false,
              },
              residencyDuration: {
                value: `allYear`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `true`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: null,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: `twelve`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
            {
              relationship: {
                value: `siblingInLaw`,
                sensitive: false,
              },
              firstName: {
                value: `Chester`,
                sensitive: false,
              },
              middleInitial: {
                value: `J`,
                sensitive: false,
              },
              lastName: {
                value: `Boone`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `112-00-2112`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `2018-07-07`,
                sensitive: false,
              },
              residencyDuration: {
                value: `sixToElevenMonths`,
                sensitive: false,
              },
              qualifyingChild: {
                value: `false`,
                sensitive: false,
              },
              eligibleDependent: {
                value: `true`,
                sensitive: false,
              },
              isClaimedDependent: {
                value: `true`,
                sensitive: false,
              },
              hohQualifyingPerson: {
                value: `false`,
                sensitive: false,
              },
              monthsLivedWithTPInUS: {
                value: null,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: null,
                sensitive: false,
              },
              scheduleEicLine4aYes: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4aNo: {
                value: `false`,
                sensitive: false,
              },
              scheduleEicLine4bYes: {
                value: `false`,
                sensitive: false,
              },
            },
          ],
          filers: [
            {
              educatorExpenses: {
                value: `200.00`,
                sensitive: false,
              },
              firstName: {
                value: `Lal`,
                sensitive: false,
              },
              middleInitial: {
                value: `J`,
                sensitive: false,
              },
              lastName: {
                value: `Boone`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `123-00-3237`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1990-07-07`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `true`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: null,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `400.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
            {
              educatorExpenses: {
                value: `0.00`,
                sensitive: false,
              },
              firstName: {
                value: `Diesel`,
                sensitive: false,
              },
              middleInitial: {
                value: `B`,
                sensitive: false,
              },
              lastName: {
                value: `Boone`,
                sensitive: false,
              },
              suffix: {
                value: null,
                sensitive: false,
              },
              tin: {
                value: `110-00-0110`,
                sensitive: true,
              },
              dateOfBirth: {
                value: `1990-05-05`,
                sensitive: false,
              },
              isPrimaryFiler: {
                value: `false`,
                sensitive: false,
              },
              ssnNotValidForEmployment: {
                value: null,
                sensitive: false,
              },
              hsaTotalDeductibleAmount: {
                value: null,
                sensitive: false,
              },
              form1099GsTotal: {
                sensitive: false,
                value: `0.00`,
              },
              interestReportsTotal: {
                sensitive: false,
                value: `800.00`,
              },
              isStudent: {
                sensitive: false,
                value: `false`,
              },
              isDisabled: {
                sensitive: false,
                value: `false`,
              },
            },
          ],
          formW2s: [
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `123-00-3237`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
            {
              unionDuesAmount: {
                value: null,
                sensitive: false,
              },
              usedTin: {
                sensitive: true,
                value: `110-00-0110`,
              },
              BOX14_NJ_UIHCWD: {
                sensitive: false,
                value: null,
              },
              BOX14_NJ_UIWFSWF: {
                sensitive: false,
                value: null,
              },
            },
          ],
          interestReports: [
            {
              has1099: {
                value: `false`,
                sensitive: false,
              },
              recipientTin: {
                value: `110-00-0110`,
                sensitive: true,
              },
              payer: {
                value: `Wells Fargo`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `800.00`,
                sensitive: false,
              },
            },
            {
              has1099: {
                value: `true`,
                sensitive: false,
              },
              recipientTin: {
                value: `123-00-3237`,
                sensitive: true,
              },
              payer: {
                value: `Fannie Mae`,
                sensitive: false,
              },
              payerTin: {
                value: null,
                sensitive: true,
              },
              '1099Amount': {
                value: `400.00`,
                sensitive: false,
              },
              taxExemptAndTaxCreditBondCusipNo: {
                value: null,
                sensitive: false,
              },
              interestOnGovernmentBonds: {
                value: `0.00`,
                sensitive: false,
              },
              taxWithheld: {
                value: `0.00`,
                sensitive: false,
              },
              taxExemptInterest: {
                value: `0.00`,
                sensitive: false,
              },
              no1099Amount: {
                value: `0.00`,
                sensitive: false,
              },
            },
          ],
        });
      });
    });
  });

  it(`should include the spouse for MFS filingStatus`, () => {
    const scenarioFileContents = readScenarioFile(`mfs-35k-cod-eitc`);
    const factJson = JSON.parse(scenarioFileContents);
    const { factGraph } = setupFactGraph(factJson.facts);

    const result = getStateExportableFactsFromGraph(factGraph);

    const { filers } = result;

    expect(Array.isArray(filers) && filers?.length).toEqual(2);
  });

  it(`should include Box 14 info`, () => {
    const scenarioFileContents = readScenarioFile(`nj-box-14`);
    const factJson = JSON.parse(scenarioFileContents);
    const { factGraph } = setupFactGraph(factJson.facts);

    const result = getStateExportableFactsFromGraph(factGraph);

    const formW2s = result.formW2s as ExportableCollection;
    const w2 = formW2s[0];
    expect(w2.BOX14_NJ_UIHCWD.value).toEqual(`160.02`);
    expect(w2.BOX14_NJ_UIWFSWF.value).toEqual(`151.41`);
  });
});

describe(getFactKeyFromPath.name, () => {
  it(`removes the leading slash for a simple key`, () => {
    const abstractPath = `/simpleKey`;
    const result = getFactKeyFromPath(abstractPath);

    expect(result).toEqual(`simpleKey`);
  });

  it(`removes slashes and converts to camel case if we have a singular concrete path key`, () => {
    const abstractPath = `/concrete/path/key`;
    const result = getFactKeyFromPath(abstractPath);

    expect(result).toEqual(`concretePathKey`);
  });

  it(`removes slashes and converts to camel case for an abstract path`, () => {
    const abstractPath = `/abstractPath/*/the/fact/key`;
    const result = getFactKeyFromPath(abstractPath);

    expect(result).toEqual(`theFactKey`);
  });
});
