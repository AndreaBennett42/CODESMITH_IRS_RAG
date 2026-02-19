import { CollectionLoop, Gate, Screen, SubSubcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  ConditionalList,
  ContextHeading,
  DFModal,
  Dollar,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
} from '../../ContentDeclarations.js';
import { Path } from '../../../fact-dictionary/Path.js';
import { ItemConfig } from '../../../components/ConditionalList/ConditionalList.js';
import { RawCondition } from '../../Condition.js';

const createCdccNotQualifyingExpensesItems = (isFiler: boolean): ItemConfig[] => {
  const qpOnlyOptions: ItemConfig[] = [
    {
      // if the qp turned 13 during the tax year (between January 2 - December 31, <TY>)
      // AND the qp was not physically or mentally unable to care for themselves
      itemKey: `afterThirteen`,
      conditions: [
        `/familyAndHousehold/*/cdccTurnedThirteenAfterJanFirst` as RawCondition,
        { operator: `isFalse`, condition: `/familyAndHousehold/*/unableToCareForSelf` },
      ],
    },
    // if the qp was physically or mentally unable to care for themselves
    { itemKey: `unableToCareQp`, conditions: [`/familyAndHousehold/*/unableToCareForSelfAndNotQcUnderAge13`] },
  ];

  const filerOnlyOptions: ItemConfig[] = [
    // if the qp was physically or mentally unable to care for themselves
    { itemKey: `unableToCareFiler`, conditions: [`/filers/*/isDisabled`] },
  ];

  const bothOptions: ItemConfig[] = [
    // if they had more than $0 in excluded benefits
    { itemKey: `depCareBenefits`, conditions: [`/cdccHasExcludedBenefits`] },
    { itemKey: `postpaidExpenses` },
    { itemKey: `futurePostpaidExpenses` },
    { itemKey: `futurePrepaidExpenses` },
  ];

  return [...(isFiler ? filerOnlyOptions : qpOnlyOptions), ...bothOptions];
};

export const makeCdccQualifyingPersonsLoop = (collection: Path, loopName: string) => {
  const isFiler = loopName === `/cdccQualifyingFilers`;
  const suffix = isFiler ? `qfiler` : `qp`;
  const collectionItemPaths = {
    qp: {
      cdccHasQualifying: `/cdccHasQualifyingNonFilers` as Path,
      cdccHasDependentCareExpenses: `/familyAndHousehold/*/cdccHasDependentCareExpenses` as Path,
      cdccHasQualifyingExpenses: `/familyAndHousehold/*/cdccHasQualifyingExpenses` as Path,
      cdccQualifyingExpenseAmount: `/familyAndHousehold/*/cdccQualifyingExpenseAmount` as Path,
      writableCdccQualifyingExpenseAmount: `/familyAndHousehold/*/writableCdccQualifyingExpenseAmount` as Path,
      cdccHadExpensesPaidToQualifyingProvider: `/familyAndHousehold/*/cdccHadExpensesPaidToQualifyingProvider` as Path,
      cdccExpensesCollected: `/familyAndHousehold/*/cdccExpensesCollected` as Path,
    },
    filer: {
      cdccHasQualifying: `/cdccHasQualifyingFilers` as Path,
      cdccHasDependentCareExpenses: `/filers/*/cdccHasDependentCareExpenses` as Path,
      cdccHasQualifyingExpenses: `/filers/*/cdccHasQualifyingExpenses` as Path,
      cdccQualifyingExpenseAmount: `/filers/*/cdccQualifyingExpenseAmount` as Path,
      writableCdccQualifyingExpenseAmount: `/filers/*/writableCdccQualifyingExpenseAmount` as Path,
      cdccHadExpensesPaidToQualifyingProvider: `/filers/*/cdccHadExpensesPaidToQualifyingProvider` as Path,
      cdccExpensesCollected: `/filers/*/cdccExpensesCollected` as Path,
    },
  };
  const qpPaths = isFiler ? collectionItemPaths.filer : collectionItemPaths.qp;

  return (
    <Gate condition={qpPaths.cdccHasQualifying}>
      <CollectionLoop
        collection={collection}
        loopName={loopName}
        autoIterate={true}
        collectionItemCompletedCondition={qpPaths.cdccExpensesCollected}
      >
        <SubSubcategory route={`qualifying-person-info-${suffix}`} headingLevel='h3'>
          <Screen route={`cdcc-qualified-expenses-${suffix}`}>
            <ContextHeading
              displayOnlyOn='edit'
              i18nKey='/heading/credits-and-deductions/credits/cdcc'
              batches={[`cdcc-0`]}
            />
            <Heading
              i18nKey={`/heading/credits-and-deductions/credits/cdcc-qualified-expenses-${suffix}`}
              batches={[`cdcc-0`]}
            />
            <DFModal
              i18nKey='/info/credits-and-deductions/credits/what-are-dependent-care-expenses'
              batches={[`cdcc-0`]}
            />
            <Boolean path={qpPaths.cdccHasDependentCareExpenses} batches={[`cdcc-3`]} />
            <SaveAndOrContinueButton />
          </Screen>
          <Gate condition={qpPaths.cdccHasDependentCareExpenses}>
            <Screen route={`cdcc-qualified-provider-${suffix}`}>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/cdcc'
                batches={[`cdcc-0`]}
              />
              <Heading
                i18nKey={`/heading/credits-and-deductions/credits/cdcc-qualified-provider-${suffix}`}
                batches={[`cdcc-0`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/credits/cdcc-not-qualified-provider'
                batches={[`cdcc-0`]}
              />
              <ConditionalList
                i18nKey='/info/credits-and-deductions/credits/cdcc-not-qualified-provider-types'
                batches={[`cdcc-0`]}
                items={[
                  { itemKey: `dependent` },
                  { itemKey: `spouse` },
                  ...(!isFiler
                    ? ([
                        {
                          itemKey: `childsParent`,
                          conditions: [`/familyAndHousehold/*/cdccCannotHaveParentAsCareProvider`],
                        },
                      ] as ItemConfig[])
                    : ([] as ItemConfig[])),
                  { itemKey: `child` },
                ]}
              />
              <Boolean path={qpPaths.cdccHadExpensesPaidToQualifyingProvider} batches={[`cdcc-2`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition={qpPaths.cdccHadExpensesPaidToQualifyingProvider}>
              <Screen route={`cdcc-qp-expenses-${suffix}`}>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/cdcc'
                  batches={[`cdcc-0`]}
                />
                <Heading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${suffix}-expenses`}
                  batches={[`cdcc-0`]}
                />
                <DFModal
                  i18nKey='/info/credits-and-deductions/credits/what-are-dependent-care-expenses'
                  batches={[`cdcc-0`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/credits/cdcc-include-prepaid-expenses'
                  batches={[`cdcc-0`]}
                />
                <ConditionalList
                  i18nKey='/info/credits-and-deductions/credits/cdcc-not-qualifying-expenses'
                  batches={[`cdcc-0`]}
                  items={createCdccNotQualifyingExpensesItems(isFiler)}
                />
                <Dollar path={qpPaths.writableCdccQualifyingExpenseAmount} batches={[`cdcc-2`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route={`cdcc-qp-transition-a-${suffix}`}>
                <ContextHeading
                  displayOnlyOn='edit'
                  i18nKey='/heading/credits-and-deductions/credits/cdcc'
                  batches={[`cdcc-0`]}
                />
                <Heading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${suffix}-transition-a`}
                  batches={[`cdcc-0`]}
                />
                <SaveAndOrContinueButton />
              </Screen>
            </Gate>
          </Gate>
          <Gate condition={{ operator: `isFalse`, condition: qpPaths.cdccHasQualifyingExpenses }}>
            <Screen route={`cdcc-qp-transition-b-${suffix}`} condition={`/cdccQpExpensesRemain`}>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey='/heading/credits-and-deductions/credits/cdcc'
                batches={[`cdcc-0`]}
              />
              <Heading
                i18nKey={`/heading/credits-and-deductions/credits/cdcc-${suffix}-transition-b`}
                batches={[`cdcc-0`]}
              />
              <InfoDisplay i18nKey='/info/credits-and-deductions/credits/cdcc-transition-b' batches={[`cdcc-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
        </SubSubcategory>
      </CollectionLoop>
    </Gate>
  );
};
