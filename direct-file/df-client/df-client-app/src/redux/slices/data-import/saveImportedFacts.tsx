/* eslint-disable no-case-declarations */
import {
  Address,
  AddressFactory,
  ConcretePath,
  DayFactory,
  DollarFactory,
  EinFactory,
  EnumFactory,
  FactGraph,
  StringFactory,
  UsPhoneNumberFactory,
} from '@irs/js-factgraph-scala';
import { AbsolutePath } from '../../../fact-dictionary/Path.js';
import { Path } from '../../../flow/Path.js';
import { FactConfig } from '../../../flow/ContentDeclarations.js';
import { getEnumOptions } from '../../../hooks/useEnumOptions.js';
import { DataImportProfile } from './dataImportProfileTypes.js';

type DataImportValue = string | Address | number | boolean | null | undefined;

// The map of the keys from the Data Import service to the fact paths to set.
const DataImportFactMap = new Map<AbsolutePath, (r: DataImportProfile, id?: string | null) => DataImportValue>([
  [
    `/importedPrimaryFilerFirstName` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        return aboutYouBasic.payload.firstName;
      }
      return ``;
    },
  ],
  [
    `/importedPrimaryFilerMiddleInitial` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        return aboutYouBasic.payload.middleInitial;
      }
      return null;
    },
  ],
  [
    `/importedPrimaryFilerLastName` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        return aboutYouBasic.payload.lastName;
      }
      return ``;
    },
  ],
  [
    `/importedPrimaryFilerDateOfBirth` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        return aboutYouBasic.payload.dateOfBirth;
      }
      return ``;
    },
  ],
  [
    `/importedPrimaryFilerAddress` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        const { mailingAddress, streetAddress, city, postalCode, stateOrProvence } = aboutYouBasic.payload;
        const importedPrimaryFilerAddress: Address = {
          streetAddress: getImportedStreetAddress(mailingAddress, streetAddress),
          city,
          postalCode,
          stateOrProvence,
          streetAddressLine2: aboutYouBasic.payload.streetAddressLine2 || ``,
          country: `USA`,
        };
        return importedPrimaryFilerAddress;
      }
      return ``;
    },
  ],
  [
    `/importedPrimaryFilerPhone` as AbsolutePath,
    (r) => {
      const { aboutYouBasic } = r.data;
      if (aboutYouBasic.state === `success`) {
        return aboutYouBasic.payload.mobileNumber;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedEmployerName` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item && item.employersAddress?.nameLine ? item.employersAddress?.nameLine : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableEmployerNameLine2` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item && item.employersAddress?.nameLine2 ? item.employersAddress?.nameLine2 : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableWages` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.wagesTipsOtherCompensation : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedEin` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.ein : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedAddressMatchesReturn` as AbsolutePath,
    (r) => {
      if (r.data.w2s.state === `success`) {
        return false;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedEmployeeAddress` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.employeeAddress : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedEmployerAddress` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.employersAddress : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableFederalWithholding` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.federalIncomeTaxWithheld : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableOasdiWages` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.socialSecurityWages : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableOasdiWithholding` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.socialSecurityTaxWithheld : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableMedicareWages` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.medicareWagesAndTips : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableMedicareWithholding` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.medicareTaxWithheld : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableOasdiTips` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.socialSecurityTips : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableAllocatedTips` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.allocatedTips : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableDependentCareBenefits` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.dependentCareBenefits : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedWritableNonQualifiedPlans` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.nonQualifiedPlans : ``;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedStatutoryEmployee` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.statutoryEmployeeIndicator : false;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedRetirementPlan` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.retirementPlanIndicator : false;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedThirdPartySickPay` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item ? item.thirdPartySickPayIndicator : false;
      }
      return ``;
    },
  ],
  [
    `/formW2s/*/importedNonstandardOrCorrectedChoice` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.w2s.state === `success`) {
        const { payload } = r.data.w2s;
        const item = payload.find((w2) => w2.id === id);
        return item && item.isStandard ? `neither` : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedHas1099` as AbsolutePath,
    (r) => {
      if (r.data.interestIncome.state === `success`) {
        return true;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedPayer` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.payerName ? item.payerName : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedPayerTin` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.payerTin ? item.payerTin : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedFactaFilingRequired` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item ? item.fatcaFilingRequirementInd : false;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox1` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box1 ? item.box1 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox2` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box2 ? item.box2 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox3` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box3 ? item.box3 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox4` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box4 ? item.box4 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox6` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box6 ? item.box6 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox8` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box8 ? item.box8 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox9` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box9 ? item.box9 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox10` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box10 ? item.box10 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox11` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box11 ? item.box11 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox12` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box12 ? item.box12 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox13` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box13 ? item.box13 : ``;
      }
      return ``;
    },
  ],
  [
    `/interestReports/*/importedBox14` as AbsolutePath,
    (r, id?: string | null) => {
      if (r.data.interestIncome.state === `success`) {
        const { payload } = r.data.interestIncome;
        const item = payload.find((int) => int.id === id);
        return item && item.box14 ? item.box14 : ``;
      }
      return ``;
    },
  ],
]);

export const saveImportedFacts = (
  profile: DataImportProfile,
  factGraph: FactGraph,
  importedFacts: FactConfig[],
  collectionId: string | null
) => {
  importedFacts.forEach((fact: FactConfig) => {
    if (fact.props.importedPath) {
      const targetPath = Path.concretePath(fact.props.path, collectionId);
      const importedTargetPath = Path.concretePath(fact.props.importedPath, collectionId);
      // @ts-ignore
      const dataImportGetter = DataImportFactMap.get(fact.props.importedPath as AbsolutePath);
      // @ts-ignore
      const sourceValue = dataImportGetter(profile, collectionId);
      const targetFactValue = factGraph.get(targetPath);
      const importedFactValue = factGraph.get(importedTargetPath);
      // `false` is a valid value we can set to the FG for booleans so we need this OR.
      const isValidFactGraphValue = sourceValue || sourceValue === false;
      if (!isValidFactGraphValue) return;
      // Check if the imported fact already has a value before setting.
      if (importedFactValue.complete) return;

      // Check if the fact already has a value before setting.
      if (targetFactValue.complete) return;

      const targetFact = factGraph.getFact(targetPath);
      const targetFactType = Object.keys(targetFact.value)[0];
      const valueFromScalaFactory = getValueFromScalaFactory(
        targetFactType,
        sourceValue,
        targetPath,
        factGraph,
        collectionId
      );

      if (valueFromScalaFactory && valueFromScalaFactory.right !== null) {
        factGraph.set(targetPath, valueFromScalaFactory.right);
        factGraph.set(importedTargetPath, valueFromScalaFactory.right);
        // eslint-disable-next-line df-rules/no-factgraph-save
        factGraph.save();
      }
    }
  });
};

const getValueFromScalaFactory = (
  type: string,
  value: DataImportValue,
  path: ConcretePath,
  factGraph: FactGraph,
  collectionId: string | null
) => {
  switch (type) {
    case `Lgov_irs_factgraph_compnodes_AddressNode__f_expr`:
      // eslint-disable-next-line no-case-declarations
      const addressToImport = value as Address;
      if (addressToImport) {
        const { streetAddress, city, postalCode, stateOrProvence, streetAddressLine2, country } = addressToImport;
        return AddressFactory(streetAddress, city, postalCode, stateOrProvence, streetAddressLine2, country);
      }
      break;
    case `Lgov_irs_factgraph_compnodes_BooleanNode__f_expr`: {
      return { right: value };
    }
    case `Lgov_irs_factgraph_compnodes_DayNode__f_expr`:
      return DayFactory(value as string);
    case `Lgov_irs_factgraph_compnodes_EinNode__f_expr`:
      return EinFactory(value as string);
    case `Lgov_irs_factgraph_compnodes_EnumNode__f_expr`:
      // For enums, we need to look up the path of the options
      const { optionsPath: enumOptionsPath } = getEnumOptions(factGraph, Path.fromConcretePath(path), collectionId);
      return EnumFactory(value as string, enumOptionsPath);
    case `Lgov_irs_factgraph_compnodes_PhoneNumberNode__f_expr`:
      return UsPhoneNumberFactory(value as string);
    case `Lgov_irs_factgraph_compnodes_StringNode__f_expr`:
      return StringFactory(value as string);
    case `Lgov_irs_factgraph_compnodes_DollarNode__f_expr`:
      if (value) return DollarFactory(value as string);
      break;
    default:
      return null;
  }
};

function getImportedStreetAddress(mailingAddress: string | null, streetAddress: string): string {
  if (mailingAddress === `` || mailingAddress === null) {
    return streetAddress;
  }

  return `${mailingAddress} ${streetAddress}`;
}
