/* eslint-disable max-len */
import { CollectionLoop, Gate, Screen, SubSubcategory, Subcategory } from '../../flowDeclarations.js';
import {
  Boolean,
  DFModal,
  DFAlert,
  Heading,
  Enum,
  SaveAndOrContinueButton,
  SetFactAction,
  CollectionItemManager,
  InfoDisplay,
  CollectionItemReference,
  IconDisplay,
  KnockoutButton,
  Dollar,
  SummaryTable,
  LimitingString,
  GenericString,
  TaxReturnAlert,
} from '../../ContentDeclarations.js';

const distributionsSubSubCategory = (
  <SubSubcategory
    route='distributions'
    completeIf='/hasCompletedHsaDistributions'
    collectionContext='/hsaDistributions'
  >
    <Screen route='hsa-distributions-loop' hasScreenRouteOverride>
      <Heading
        i18nKey='/heading/income/hsa/distributions/intro'
        batches={[`hsa-0`]}
        conditions={[
          { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasHsaDistributions` },
        ]}
      />
      <Heading
        i18nKey='/heading/income/hsa/distributions/intro-spouse'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { operator: `isFalseOrIncomplete`, condition: `/hasHsaDistributions` },
        ]}
      />
      <Heading i18nKey='/heading/income/hsa/distributions' batches={[`hsa-1`]} condition={`/hasHsaDistributions`} />
      <InfoDisplay
        i18nKey='/info/income/hsa/distributions/add'
        batches={[`hsa-0`]}
        condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/distributions/add-spouse'
        batches={[`hsa-0`]}
        condition='/isFilingStatusMFJ'
      />
      <DFModal
        i18nKey='/info/income/hsa/distributions/supported-distributions-modal'
        batches={[`hsa-0`]}
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasHsaDistributions` }}
      />
      <DFModal
        i18nKey='/info/income/hsa/distributions/supported-distributions-alert-modal'
        batches={[`hsa-0`]}
        condition={`/hasHsaDistributions`}
      />
      <DFModal
        i18nKey='/info/income/hsa/how-taxable-distributions-calculated-modal'
        batches={[`hsa-0`]}
        condition={`/hasHsaDistributions`}
      />
      <DFAlert
        i18nKey='/info/income/hsa/distributions/supported-distributions-alert'
        headingLevel='h3'
        type='info'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/hasHsaDistributions` }}
        batches={[`hsa-0`]}
      >
        <DFModal i18nKey='/info/income/hsa/distributions/supported-distributions-alert-modal' batches={[`hsa-0`]} />
      </DFAlert>
      <CollectionItemManager path='/hsaDistributions' loopName='/hsaDistributions' donePath='/hsaDistributionsIsDone' />
    </Screen>
    <CollectionLoop
      loopName='/hsaDistributions'
      collection='/hsaDistributions'
      collectionItemCompletedCondition='/hsaDistributions/*/isComplete'
      isInner={true}
      donePath='/hsaDistributionsIsDone'
      dataViewSections={[
        {
          i18nKey: `dataviews./flow/income/hsa/distributions.primaryFiler`,
          condition: `/hsaDistributions/*/filer/isPrimaryFiler`,
        },
        {
          i18nKey: `dataviews./flow/income/hsa/distributions.secondaryFiler`,
          condition: `/hsaDistributions/*/filer/isSecondaryFiler`,
        },
      ]}
    >
      <SubSubcategory route='distribution-info'>
        {/* We probably need to take a second pass on the conditional here (it could be just spouse has HSAs or vice-versa with MFJ status -- so we can't just use a placeholder) -- maybe we can filter which filers are displayed or preset the filer based on /filersWithHsa */}
        <Screen route='hsa-whose-distribution' condition='/isFilingStatusMFJ'>
          <TaxReturnAlert
            i18nKey='/info/income/hsa/secondary-filer-income-without-mfj'
            headingLevel='h3'
            type='error'
            condition='/hsaDistributions/*/secondaryFilerUsedWithoutMFJ'
            batches={[`hsa-1`]}
          />
          <TaxReturnAlert
            i18nKey='/info/income/hsa/distributions-without-hsa-activity'
            headingLevel='h3'
            type='error'
            condition='/hsaDistributions/*/hasDistributionsWithoutHsaActivity'
            batches={[`hsa-1`]}
          />
          <Heading i18nKey='/heading/income/hsa/distribution/add-whose' batches={[`hsa-0`]} />
          <CollectionItemReference path='/hsaDistributions/*/filer' displayOnlyOn='edit' />
          <GenericString path='/hsaDistributions/*/filer/fullName' displayOnlyOn='data-view' batches={[`hsa-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='hsa-add-trustee-name'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-trustee' batches={[`hsa-0`]} />
          <SetFactAction
            path='/hsaDistributions/*/filer'
            source='/primaryFiler'
            conditions={[
              { operator: `isIncomplete`, condition: `/hsaDistributions/*/filer` },
              { operator: `isFalseOrIncomplete`, condition: `/treatAsMFJ` },
            ]}
          />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-trustee' batches={[`hsa-0`]} />
          <LimitingString path='/hsaDistributions/*/writableTrusteeName' batches={[`hsa-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='hsa-distributions-add-box1'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-box-1' batches={[`hsa-0`]} />
          <Dollar path='/hsaDistributions/*/writableGrossDistribution' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='hsa-distributions-rollovers'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-rollovers' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-rollovers' batches={[`hsa-0`]} />
          <Boolean path='/hsaDistributions/*/writableDistributionsRolloverBool' batches={[`hsa-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='hsa-distributions-rollovers-amount'
          condition={{
            operator: `isTrue`,
            condition: `/hsaDistributions/*/writableDistributionsRolloverBool`,
          }}
        >
          <TaxReturnAlert
            type='error'
            i18nKey='/info/income/hsa/distributions/hsa-distributions-excess-rollover-amount'
            condition='/hsaDistributions/*/hasRolloverAmountMoreThanGrossDistributions'
            batches={[`hsa-1`]}
          />
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-rollovers-amount' batches={[`hsa-0`]} />
          <Dollar path='/hsaDistributions/*/writableDistributionsRolloverAmount' required={false} batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-distributions-excess-withdrawn'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-distributions-excess-withdrawn' batches={[`hsa-0`]} />
          <InfoDisplay
            i18nKey='/info/income/hsa/distributions/hsa-distributions-excess-withdrawn'
            batches={[`hsa-0`]}
          />
          <DFModal
            i18nKey='/info/income/hsa/distributions/hsa-distributions-excess-withdrawn-info-modal'
            batches={[`hsa-0`]}
          />
          <Boolean path='/hsaDistributions/*/hasWithdrawnExcessContributions' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen
          route='hsa-distributions-excess-withdrawn-amount'
          condition='/hsaDistributions/*/hasWithdrawnExcessContributions'
        >
          <Heading
            i18nKey='/heading/income/hsa/distributions/hsa-distributions-excess-withdrawn-amount'
            batches={[`hsa-0`]}
          />
          <TaxReturnAlert
            type='error'
            i18nKey='/info/income/hsa/distributions/hsa-distributions-excess-withdrawn-amount-is-too-much'
            condition='/hsaDistributions/*/hasWithdrawnMoreThanGrossDistributions'
            batches={[`hsa-0`]}
          />

          <Dollar
            path='/hsaDistributions/*/writableWithdrawnExcessContributionsAmount'
            required={false}
            batches={[`hsa-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-distributions-add-box2'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-box-2' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-box-2' batches={[`hsa-0`]} />
          <Dollar
            path='/hsaDistributions/*/writableEarningsOnExcessContributions'
            required={false}
            batches={[`hsa-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='hsa-distributions-box2-ko'
          condition='/flowKnockoutEarningsOnExcessContributions'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-ko-5329-needed' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-distributions-box2-ko' batches={[`hsa-1`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='hsa-distributions-add-box3'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-distributions-add-box-3' batches={[`hsa-0`]} />
          <Enum path='/hsaDistributions/*/hsaDistributionCode' renderAs='select' batches={[`hsa-1`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='hsa-distributions-box3-ko' condition='/flowKnockoutHsaDistributionCode' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/hsa-distribution-code' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-distributions-box3-ko' batches={[`hsa-0`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='hsa-distributions-add-box4'>
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-distributions-add-box-4' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-distributions-add-box-4' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/distributions/about-fair-market-value-modal' batches={[`hsa-0`]} />
          <Dollar path='/hsaDistributions/*/writableFmvOnDateOfDeath' required={false} batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='hsa-ko-fmv' condition='/flowKnockoutHsaDistributionFMV' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-ko-fmv' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-ko-fmv' batches={[`hsa-0`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
        <Screen route='hsa-distributions-add-medical-expenses'>
          <Heading
            i18nKey='/heading/income/hsa/distributions/hsa-distributions-add-medical-expenses'
            batches={[`hsa-0`]}
          />
          <DFModal
            i18nKey='/info/income/hsa/distributions/hsa-distributions-add-medical-expenses-info-modal'
            batches={[`hsa-0`]}
          />
          <DFModal
            i18nKey='/info/income/hsa/distributions/hsa-distributions-add-medical-expenses-qualified-modal'
            batches={[`hsa-0`]}
          />
          <Dollar path='/hsaDistributions/*/writableQualifiedMedExpenses' batches={[`hsa-1`]} />
          <SetFactAction path='/hsaDistributions/*/hasSeenLastAvailableScreen' source='/flowTrue' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen
          route='hsa-ko-unqualified-distributions'
          condition='/flowKnockoutHsaUnqualifiedDistributions'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/income/hsa/distributions/hsa-ko-unqualified-distributions' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/distributions/hsa-ko-unqualified-distributions' batches={[`hsa-0`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </SubSubcategory>
    </CollectionLoop>
  </SubSubcategory>
);

const coverageAndContributions = (
  <CollectionLoop
    collection='/filersWithHsa'
    loopName='/filersWithHsa'
    autoIterate={true}
    collectionItemCompletedCondition='/filers/*/hasCompletedCoverageAndContribution'
  >
    <SubSubcategory route='coverage-and-contributions'>
      <Screen route='hsa-breather-about-you'>
        <Heading i18nKey='/heading/income/hsa/hsa-breather' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/hsa-breather' batches={[`hsa-0`]} condition='/filers/*/isPrimaryFiler' />
        <InfoDisplay
          i18nKey='/info/income/hsa/hsa-breather-spouse'
          batches={[`hsa-0`]}
          condition='/filers/*/isSecondaryFiler'
        />
        <InfoDisplay
          i18nKey='/info/income/hsa/hsa-breather-mfj-spouse-info'
          batches={[`hsa-0`]}
          conditions={[`/filers/*/isPrimaryFiler`, `/bothFilersHaveHsa`]}
        />
        <SaveAndOrContinueButton />
      </Screen>
      <Gate condition='/filers/*/isPrimaryFiler'>
        <Screen
          route='hsa-contributions-additional-y-n-primary'
          condition={{
            operator: `isFalse`,
            condition: `/hasPrimaryFilerHsaContributionsFromW2s`,
          }}
        >
          <Heading i18nKey='/heading/income/hsa/hsa-contributions-additional-y-n-primary' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-additional-y-n-primary' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-information-modal' batches={[`hsa-1`]} />
          <Boolean path='/filers/*/writablePrimaryFilerHasMadeContributionsToHsa' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
      <Gate condition='/filers/*/isSecondaryFiler'>
        <Screen
          route='hsa-contributions-additional-y-n-secondary'
          condition='/flowSecondaryFilerHasNoHsaContributionsOnW2andMFJ'
        >
          <Heading i18nKey='/heading/income/hsa/hsa-contributions-additional-y-n-secondary' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-additional-y-n-secondary' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-information-modal' batches={[`hsa-1`]} />
          <Boolean path='/filers/*/writableSecondaryFilerHasMadeContributionsToHsa' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>

      <Screen
        route='hsa-contributions-additional-summary-none-allowed-ko'
        condition='/flowKnockoutFilerIsDependentAndContributesToHsa'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading
          i18nKey='/heading/knockout/hsa-contributions-summary-none-allowed'
          batches={[`hsa-0`]}
          condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
        />
        <Heading
          i18nKey='/heading/knockout/hsa-contributions-summary-none-allowed-spouse'
          batches={[`hsa-1`]}
          condition='/isFilingStatusMFJ'
        />
        <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-what-makes-eligible' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-how-limit-calculated' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/hsa-contributions-summary-another-tool' batches={[`hsa-0`]} />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>

      {/* Coverage section */}
      <Gate
        condition={{
          operator: `isFalse`,
          condition: `/filers/*/flowSkipToHsaTestingPeriodContributionCheck`,
        }}
      >
        <Screen route='hsa-coverage-breather'>
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-breather'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-breather-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-what-makes-eligible' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-how-limit-calculated' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-coverage-medicare-status'>
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-medicare-status'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-medicare-status-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/hsa-why-ask-medicare-enrollment' batches={[`hsa-0`]} />
          <Enum
            condition='/filers/*/isPrimaryFiler'
            i18nKeySuffixContext='self'
            path='/filers/*/enrolledInMedicare'
            batches={[`hsa-0`]}
          />
          <Enum
            condition='/filers/*/isSecondaryFiler'
            i18nKeySuffixContext='mfj'
            path='/filers/*/enrolledInMedicare'
            batches={[`hsa-0`]}
          />
          <SaveAndOrContinueButton />
        </Screen>

        {/* We have to add a conditional on the screen for the previous knockout (otherwise this screen will show first even if they were knocked out above) */}
        <Screen
          route='hsa-coverage-status'
          condition={{ operator: `isFalse`, condition: `/flowKnockoutHdhpCoverageStatusNotNeeded` }}
        >
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-status'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-status-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/hsa-hdhp-requirements' batches={[`hsa-0`]} />
          <Enum path='/filers/*/hsaHdhpCoverageStatus' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Gate condition='/filers/*/flowShouldAskForHsaHdhpCoverageType'>
          <Screen route='hsa-coverage-add-type'>
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-add-type'
              batches={[`hsa-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-add-type-spouse'
              batches={[`hsa-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <InfoDisplay
              i18nKey='/info/income/hsa/hsa-coverage-add-type'
              batches={[`hsa-0`]}
              condition='/isFilingStatusSingle'
            />
            <DFModal
              i18nKey='/info/income/hsa/hsa-coverage-add-type-mfj-mfs-with-spouse'
              batches={[`hsa-1`]}
              condition='/isFilingStatusMfjOrMfs'
            />
            <Enum path='/filers/*/typeOfHdhp' batches={[`hsa-0`]} />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='hsa-coverage-other-y-n'
            condition='/filers/*/flowShouldAskHadCoverageIneligibleForHSAContributions'
          >
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-other-y-n'
              batches={[`hsa-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-other-y-n-spouse'
              batches={[`hsa-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <DFModal i18nKey='/info/income/hsa/hsa-coverage-other-y-n' batches={[`hsa-0`]} />
            <Enum path='/filers/*/hadOtherCoverageIneligibleForHSA' batches={[`hsa-0`]} />
            <SaveAndOrContinueButton />
          </Screen>

          <Gate condition='/filers/*/flowShowHsaMfsLine6Check'>
            <Screen route='hsa-coverage-married-not-mfj-line-6-check'>
              <Heading i18nKey='/heading/income/hsa/hsa-coverage-married-not-mfj-line-6-check' batches={[`hsa-0`]} />
              <DFModal i18nKey='/info/income/hsa/hsa-coverage-married-not-mfj-line-6-check' batches={[`hsa-0`]} />
              <Boolean path='/filers/*/writableMfsLine6Check' batches={[`hsa-0`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Screen
              route='hsa-coverage-married-not-mfj-line-6-ko'
              condition='/flowKnockoutMFSLine6Check'
              isKnockout={true}
            >
              <IconDisplay name='ErrorOutline' size={9} isCentered />
              <Heading i18nKey='/heading/knockout/hsa-coverage-line-6-ko' batches={[`hsa-0`]} />
              <InfoDisplay i18nKey='/info/knockout/hsa-coverage-MFS-line-6-ko' batches={[`hsa-0`]} />
              <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
              <KnockoutButton i18nKey='button.knockout' />
            </Screen>
          </Gate>
        </Gate>

        <Screen route='hsa-coverage-ko' condition='/flowKnockoutHsaCoverage' isKnockout={true}>
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/hsa-coverage-changing' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/knockout/hsa-coverage-ko' batches={[`hsa-1`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>

        <Screen
          route='hsa-contributions-summary-none-allowed-ko'
          condition='/flowKnockoutContributionSummaryNoneAllowed'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading
            i18nKey='/heading/knockout/hsa-contributions-summary-none-allowed'
            batches={[`hsa-0`]}
            condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
          />
          <Heading
            i18nKey='/heading/knockout/hsa-contributions-summary-none-allowed-spouse'
            batches={[`hsa-1`]}
            condition='/isFilingStatusMFJ'
          />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-what-makes-eligible' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-how-limit-calculated' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/income/hsa/hsa-contributions-summary-another-tool' batches={[`hsa-0`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>

        <Screen
          route='hsa-coverage-marital-change'
          condition='/filers/*/flowShouldAskForChangeInMaritalStatusDuringTaxYear'
        >
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-marital-change'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/hsa-coverage-marital-change-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <InfoDisplay i18nKey='/info/income/hsa/hsa-coverage-marital-change' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-coverage-marital-change-modal' batches={[`hsa-0`]} />
          <Enum path='/filers/*/writableChangeInMaritalStatusDuringTaxYear' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Gate condition='/filers/*/flowShouldAskIfChangeInMaritalStatusAffectsContributionLimit'>
          <Screen route='hsa-coverage-line-6'>
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-line-6'
              batches={[`hsa-0`]}
              condition='/filers/*/isPrimaryFiler'
            />
            <Heading
              i18nKey='/heading/income/hsa/hsa-coverage-line-6-spouse'
              batches={[`hsa-0`]}
              condition='/filers/*/isSecondaryFiler'
            />
            <DFModal i18nKey='/info/income/hsa/hsa-coverage-line-6-modal' batches={[`hsa-0`]} />
            <Boolean path='/filers/*/writableMaritalChangeAffectContributionLimitBool' batches={[`hsa-0`]} />
            <SaveAndOrContinueButton />
          </Screen>

          <Screen
            route='hsa-coverage-line-6-ko'
            condition='/flowKnockoutMaritalChangeAffectsContributionLimit'
            isKnockout={true}
          >
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/hsa-coverage-line-6-ko' batches={[`hsa-0`]} />
            <InfoDisplay i18nKey='/info/income/hsa/coverage/hsa-coverage-line-6-ko' batches={[`hsa-0`]} />
            <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
        </Gate>

        {/* Contributions section */}
        <Screen route='hsa-contributions-breather'>
          <Heading i18nKey='/heading/income/hsa/hsa-contributions-breather' batches={[`hsa-0`]} />
          <InfoDisplay
            i18nKey='/info/income/hsa/hsa-contributions-breather'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <InfoDisplay
            i18nKey='/info/income/hsa/hsa-contributions-breather-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-breather-multiple-owners' />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-contributions-add-contributions-made-ty'>
          <Heading
            i18nKey='/heading/income/hsa/add-contributions-made-ty'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/add-contributions-made-ty-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/contributions-do-not-include' batches={[`hsa-0`]} />
          <DFModal i18nKey='/info/income/hsa/hsa-contributions-information-modal' batches={[`hsa-1`]} />
          <Dollar path='/filers/*/writableHsaNonemployerContributionsTaxYear' batches={[`hsa-0`]} required={false} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-contributions-add-contributions-made-ty+1'>
          <Heading
            i18nKey='/heading/income/hsa/add-contributions-made-ty+1'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/add-contributions-made-ty+1-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <InfoDisplay i18nKey='/info/income/hsa/add-contributions-made-ty+1' batches={[`hsa-0`]} />
          <InfoDisplay
            i18nKey='/info/income/hsa/add-contributions-made-ty+1-line2'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <InfoDisplay
            i18nKey='/info/income/hsa/add-contributions-made-ty+1-line2-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <DFModal i18nKey='/info/income/hsa/contributions-do-not-include' batches={[`hsa-0`]} />
          <Dollar
            path='/filers/*/writableHsaNonemployerContributionsTaxYearPlusOne'
            batches={[`hsa-0`]}
            required={false}
          />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-contributions-qualified-funding-distribution'>
          <Heading
            i18nKey='/heading/income/hsa/qualified-funding-distribution'
            batches={[`hsa-0`]}
            condition='/filers/*/isPrimaryFiler'
          />
          <Heading
            i18nKey='/heading/income/hsa/qualified-funding-distribution-spouse'
            batches={[`hsa-0`]}
            condition='/filers/*/isSecondaryFiler'
          />
          <InfoDisplay i18nKey='/info/income/hsa/qualified-funding-distribution' batches={[`hsa-0`]} />
          <Boolean path='/filers/*/hasMadeQualifiedHsaFundingDistribution' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen
          route='hsa-ko-qualified-hsa-funding-distribution'
          condition='/flowKnockoutMadeQualifiedHsaFundingDistribution'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/hsa-qualified-funding-distribution' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/knockout/hsa-qualified-funding-distribution' batches={[`hsa-0`]} />
          <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </Gate>
      <Screen
        route='hsa-contributions-summary-excess-KO'
        condition='/flowHsaContributionsOverLimitKnockout'
        isKnockout={true}
      >
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/income/hsa/hsa-contributions-over-ko' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-what-makes-eligible' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-how-limit-calculated' batches={[`hsa-0`]} />
        <SummaryTable
          i18nKey='/info/income/hsa/hsa-contributions-summary-over'
          batches={[`hsa-0`]}
          conditions={[
            `/primaryFiler/hasMadeContributionsToHsa`,
            {
              operator: `isFalse`,
              condition: `/flowHsaContributionsSummaryTableOverageVisibility`,
            },
          ]}
          items={[
            {
              itemKey: `hsaContributions`,
            },
            {
              itemKey: `hsaContributionsTotal`,
            },
          ]}
        />
        <SummaryTable
          i18nKey='/info/income/hsa/hsa-contributions-summary-over'
          batches={[`hsa-0`]}
          conditions={[`/primaryFiler/hasMadeContributionsToHsa`, `/flowHsaContributionsSummaryTableOverageVisibility`]}
          items={[
            {
              itemKey: `hsaContributions`,
            },
            {
              itemKey: `hsaContributionsTotal`,
            },
            {
              itemKey: `hsaContributionsOverage`,
            },
          ]}
        />
        <SummaryTable
          i18nKey='/info/income/hsa/hsa-contributions-summary-over'
          batches={[`hsa-0`]}
          conditions={[
            `/secondaryFiler/hasMadeContributionsToHsa`,
            {
              operator: `isFalse`,
              condition: `/flowHsaContributionsSummaryTableOverageVisibilitySpouse`,
            },
          ]}
          items={[
            {
              itemKey: `hsaContributionsSpouse`,
            },
            {
              itemKey: `hsaContributionsTotalSpouse`,
            },
          ]}
        />
        <SummaryTable
          i18nKey='/info/income/hsa/hsa-contributions-summary-over'
          batches={[`hsa-0`]}
          conditions={[
            `/secondaryFiler/hasMadeContributionsToHsa`,
            `/flowHsaContributionsSummaryTableOverageVisibilitySpouse`,
          ]}
          items={[
            {
              itemKey: `hsaContributionsSpouse`,
            },
            {
              itemKey: `hsaContributionsTotalSpouse`,
            },
            {
              itemKey: `hsaContributionsTotalOverageSpouse`,
            },
          ]}
        />
        <InfoDisplay i18nKey='/info/income/hsa/hsa-contributions-summary-over-info' batches={[`hsa-0`]} />
        <DFAlert i18nKey='/info/knockout/generic-other-ways-to-file' headingLevel='h2' type='warning' />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>
  </CollectionLoop>
);

export const HsaSubcategory = (
  <Subcategory
    route='hsa'
    completeIf='/isHsaSectionComplete'
    dataItems={[
      {
        itemKey: `hsaTaxableIsZero`,
        conditions: [`/requiresForm8889`],
      },
      {
        itemKey: `hsaNoneReported`,
        conditions: [{ operator: `isFalse`, condition: `/requiresForm8889` }],
      },
    ]}
  >
    <Screen route='hsa-intro'>
      <Heading i18nKey='/heading/income/hsa' batches={[`hsa-0`]} />
      <DFModal
        i18nKey='/info/income/hsa/intro'
        batches={[`hsa-0`]}
        conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
      />
      <DFModal
        i18nKey='/info/income/hsa/intro-mfj'
        batches={[`hsa-0`]}
        conditions={[{ condition: `/isFilingStatusMFJ` }]}
      />
      <DFModal i18nKey='/info/income/hsa/add' batches={[`hsa-0`]} />
      <InfoDisplay
        i18nKey='/info/income/hsa/intro-contributions'
        batches={[`hsa-0`]}
        conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/intro-contributions-mfj'
        batches={[`hsa-0`]}
        conditions={[{ condition: `/isFilingStatusMFJ` }]}
      />
      <DFAlert
        i18nKey='/info/income/hsa/supported-income-alert'
        headingLevel='h3'
        type='info'
        condition={{ operator: `isFalseOrIncomplete`, condition: `/isHsaSectionComplete` }}
        batches={[`hsa-0`]}
      >
        <DFModal i18nKey='/info/income/hsa/supported-income-alert-modal' batches={[`hsa-0`]} />
      </DFAlert>
      <SaveAndOrContinueButton />
    </Screen>
    <Screen route='hsa-already-reported-w2-contributions' condition='/hasHsaContributionsOnW2'>
      <Heading i18nKey='/heading/income/hsa/exisiting-hsa-contributions' batches={[`hsa-0`]} />
      <InfoDisplay
        i18nKey='/info/income/hsa/exisiting-hsa-contributions-primary'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
          { operator: `isFalse`, condition: `/mfjBothFilersHaveHsaContributionsOnW2` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/exisiting-hsa-contributions-mfj-spouse'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { condition: `/hasSecondaryFilerHsaContributionsFromW2s` },
          { operator: `isFalse`, condition: `/mfjBothFilersHaveHsaContributionsOnW2` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/exisiting-hsa-contributions-mfj-both'
        batches={[`hsa-0`]}
        conditions={[{ condition: `/mfjBothFilersHaveHsaContributionsOnW2` }]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/contribution-amount-primary'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
          { operator: `isFalse`, condition: `/mfjBothFilersHaveHsaContributionsOnW2` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/contribution-amount-spouse'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/isFilingStatusMFJ` },
          { condition: `/hasSecondaryFilerHsaContributionsFromW2s` },
          { operator: `isFalse`, condition: `/mfjBothFilersHaveHsaContributionsOnW2` },
        ]}
      />
      <InfoDisplay
        i18nKey='/info/income/hsa/contribution-amount-mfj'
        batches={[`hsa-0`]}
        conditions={[{ condition: `/mfjBothFilersHaveHsaContributionsOnW2` }]}
      />
      <DFModal i18nKey='/info/income/hsa/contribution-ty-minus-1' batches={[`hsa-0`]} />
      <DFModal i18nKey='/info/income/hsa/contribution-ty-plus-1' batches={[`hsa-0`]} />
      <SaveAndOrContinueButton />
    </Screen>

    <SubSubcategory route='hsa-intro'>
      <Screen route='hsa-y-n' condition='/flowShowHsaYesNo'>
        <Heading
          i18nKey='/heading/income/hsa/yes-no-primary'
          batches={[`hsa-0`]}
          conditions={[
            { operator: `isFalse`, condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
            { operator: `isFalse`, condition: `/isFilingStatusMFJ` },
          ]}
        />
        <Heading
          i18nKey='/heading/income/hsa/yes-no-primary'
          batches={[`hsa-0`]}
          conditions={[
            { operator: `isFalse`, condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
            { condition: `/hasSecondaryFilerHsaContributionsFromW2s` },
            { condition: `/isFilingStatusMFJ` },
          ]}
        />
        <Heading
          i18nKey='/heading/income/hsa/yes-no-secondary'
          batches={[`hsa-0`]}
          conditions={[
            { condition: `/isFilingStatusMFJ` },
            { condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
            { operator: `isFalse`, condition: `/hasSecondaryFilerHsaContributionsFromW2s` },
          ]}
        />
        <Heading
          i18nKey='/heading/income/hsa/yes-no-both'
          batches={[`hsa-0`]}
          conditions={[
            { condition: `/isFilingStatusMFJ` },
            { operator: `isFalse`, condition: `/hasPrimaryFilerHsaContributionsFromW2s` },
            { operator: `isFalse`, condition: `/hasSecondaryFilerHsaContributionsFromW2s` },
          ]}
        />
        <Boolean path='/someFilerHadNonW2HsaActivity' batches={[`hsa-0`]} />
        <SaveAndOrContinueButton />
      </Screen>

      <Screen route='hsa-activity-primary-y-n' condition='/flowShowHsaPrimaryYesNo'>
        <Heading i18nKey='/heading/income/hsa/activity-yes-no-primary' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/activity-yes-no-primary' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/can-an-hsa-have-multiple-owners' batches={[`hsa-0`]} />
        <Boolean path='/writablePrimaryFilerHadNonW2HsaActivity' batches={[`hsa-0`]} />
        <SaveAndOrContinueButton />
      </Screen>

      <Screen route='hsa-activity-secondary-y-n' condition='/flowShowHsaSecondaryYesNo'>
        <Heading i18nKey='/heading/income/hsa/activity-yes-no-secondary' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/activity-yes-no-secondary' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/can-an-hsa-have-multiple-owners' batches={[`hsa-0`]} />
        <Boolean path='/writableSecondaryFilerHadNonW2HsaActivity' batches={[`hsa-0`]} />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='hsa-add-account-type' condition='/someFilerHasSomeHsaActivityNotConsideringKos'>
        <Heading
          i18nKey='/heading/income/hsa/medical-savings-account-type'
          batches={[`hsa-0`]}
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Heading
          i18nKey='/heading/income/hsa/medical-savings-account-type-mfj'
          batches={[`hsa-0`]}
          conditions={[{ condition: `/isFilingStatusMFJ` }]}
        />
        <InfoDisplay i18nKey='/info/income/hsa/medical-savings-account-type' batches={[`hsa-0`]} />
        <DFModal
          i18nKey='/info/income/hsa/medical-savings-account-type-no'
          batches={[`hsa-0`]}
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <DFModal
          i18nKey='/info/income/hsa/medical-savings-account-type-no-mfj'
          batches={[`hsa-0`]}
          conditions={[{ condition: `/isFilingStatusMFJ` }]}
        />
        <Boolean
          path='/hasHsaMedicalSavingsAccountType'
          i18nKeySuffixContext='i'
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Boolean
          path='/hasHsaMedicalSavingsAccountType'
          i18nKeySuffixContext='we'
          conditions={[{ condition: `/isFilingStatusMFJ` }]}
        />
        <SaveAndOrContinueButton />
      </Screen>

      <Screen route='hsa-excess-contributions-previous-year' condition='/flowToExcessContributionsPreviousYear'>
        <Heading
          i18nKey='/heading/income/hsa/hsa-excess-contributions-previous-year'
          batches={[`hsa-0`]}
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Heading
          i18nKey='/heading/income/hsa/hsa-excess-contributions-previous-year-mfj'
          batches={[`hsa-0`]}
          conditions={[{ condition: `/isFilingStatusMFJ` }]}
        />
        <InfoDisplay i18nKey='/info/income/hsa/hsa-excess-contributions-previous-year' batches={[`hsa-0`]} />
        <Boolean path='/writableHasHsaExcessContributionsPreviousYear' />
        <SaveAndOrContinueButton />
      </Screen>
      <Screen route='hsa-excess-contributions-y-n' condition='/flowToHsaExcessContributions'>
        <Heading
          i18nKey='/heading/income/hsa/hsa-excess-contributions-y-n'
          batches={[`hsa-0`]}
          conditions={[{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }]}
        />
        <Heading
          i18nKey='/heading/income/hsa/hsa-excess-contributions-y-n-mfj'
          batches={[`hsa-0`]}
          conditions={[{ condition: `/isFilingStatusMFJ` }]}
        />
        <InfoDisplay i18nKey='/info/income/hsa/hsa-excess-contributions-y-n' batches={[`hsa-0`]} />
        <DFModal i18nKey='/info/income/hsa/where-do-i-find-this-information' batches={[`hsa-0`]} />
        <Boolean path='/writableHasHsaWithdrawnExcessContributionsYesNo' />
        <SaveAndOrContinueButton />
      </Screen>

      <Screen route='hsa-ko-excess-contributions' condition='/flowKnockoutHsaExcessContributions' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/hsa/excess-contributions' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/excess-contributions-not-supported' batches={[`hsa-0`]} />
        <DFAlert
          i18nKey='/info/knockout/generic-other-ways-to-file'
          headingLevel='h2'
          type='warning'
          batches={[`hsa-0`]}
        />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>

      <Screen route='hsa-ko-8853-needed' condition='/flowKnockoutHsaMedicalSavingsAccountType' isKnockout={true}>
        <IconDisplay name='ErrorOutline' size={9} isCentered />
        <Heading i18nKey='/heading/knockout/hsa/medical-savings-account-type' batches={[`hsa-0`]} />
        <InfoDisplay i18nKey='/info/income/hsa/medical-savings-account-type-ko' batches={[`hsa-0`]} />
        <DFAlert
          i18nKey='/info/knockout/generic-other-ways-to-file'
          headingLevel='h2'
          type='warning'
          batches={[`hsa-0`]}
        />
        <KnockoutButton i18nKey='button.knockout' />
      </Screen>
    </SubSubcategory>

    {/* contributions and distributions */}
    <Gate condition='/someFilerHasSomeHsaActivity'>{coverageAndContributions}</Gate>

    <SubSubcategory route='hsa-testing-period-check'>
      <Gate condition='/flowShowHsaTestingPeriodContributionCheck'>
        <Screen route='hsa-testing-period-contribution-check'>
          <Heading
            i18nKey='/heading/income/hsa/testing-period-contribution-primary'
            batches={[`hsa-0`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/primaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Heading
            i18nKey='/heading/income/hsa/testing-period-contribution-spouse'
            batches={[`hsa-0`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/secondaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Heading
            i18nKey='/heading/income/hsa/testing-period-contribution-both'
            batches={[`hsa-0`]}
            condition='/bothFilersHaveToShowTestingPeriodCheck'
          />
          <Boolean
            path='/someFilersMadeTestingPeriodContribution'
            batches={[`hsa-0`]}
            i18nKeySuffixContext='primary'
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/primaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Boolean
            path='/someFilersMadeTestingPeriodContribution'
            batches={[`hsa-0`]}
            i18nKeySuffixContext='secondary'
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/secondaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Boolean
            path='/someFilersMadeTestingPeriodContribution'
            batches={[`hsa-0`]}
            i18nKeySuffixContext='both'
            condition='/bothFilersHaveToShowTestingPeriodCheck'
          />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen route='hsa-testing-period-additional-income' condition='/flowHsaTestingPeriodCheckYes'>
          <Heading
            i18nKey='/heading/income/hsa/testing-period-additional-income-primary'
            batches={[`hsa-1`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/primaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Heading
            i18nKey='/heading/income/hsa/testing-period-additional-income-spouse'
            batches={[`hsa-1`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/secondaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <Heading
            i18nKey='/heading/income/hsa/testing-period-additional-income-both'
            batches={[`hsa-1`]}
            condition='/bothFilersHaveToShowTestingPeriodCheck'
          />
          <DFModal
            i18nKey='/info/income/hsa/testing-period-additional-income-primary'
            batches={[`hsa-0`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/primaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <DFModal
            i18nKey='/info/income/hsa/testing-period-additional-income-spouse'
            batches={[`hsa-0`]}
            conditions={[
              { operator: `isFalse`, condition: `/bothFilersHaveToShowTestingPeriodCheck` },
              { operator: `isTrue`, condition: `/secondaryFiler/hasToShowTestingPeriodCheck` },
            ]}
          />
          <DFModal
            i18nKey='/info/income/hsa/testing-period-additional-income-both'
            batches={[`hsa-0`]}
            condition='/bothFilersHaveToShowTestingPeriodCheck'
          />
          <Boolean path='/someFilersHaveTestingPeriodAdditionalIncome' batches={[`hsa-0`]} />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen
          route='hsa-testing-period-income-ko'
          condition='/flowKnockoutHsaTestingPeriodAdditionalIncomeYes'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/hsa/testing-period-additional-income' batches={[`hsa-0`]} />
          <InfoDisplay i18nKey='/info/knockout/hsa-testing-period-additional-income' batches={[`hsa-0`]} />
          <DFAlert
            i18nKey='/info/knockout/generic-other-ways-to-file'
            headingLevel='h2'
            type='warning'
            batches={[`hsa-0`]}
          />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>
      </Gate>
    </SubSubcategory>

    <Screen route='hsa-contributions-summary-under' condition='/someFilersHaveHsaContributionsNotExceedLimits'>
      <Heading i18nKey='/heading/income/hsa/hsa-contributions-summary-under' batches={[`hsa-0`]} />
      <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-what-makes-eligible' batches={[`hsa-0`]} />
      <DFModal i18nKey='/info/income/hsa/hsa-contributions-summary-how-limit-calculated' batches={[`hsa-0`]} />
      <SummaryTable
        i18nKey='/info/income/hsa/hsa-contributions-summary-under'
        batches={[`hsa-0`]}
        conditions={[
          { condition: `/primaryFiler/hasHsaContributionsNotExceedLimits` },
          { operator: `isTrueOrIncomplete`, condition: `/primaryFiler/primaryFilerHasMadeContributionsToHsa` },
        ]}
        items={[
          {
            itemKey: `hsaContributions`,
          },
          {
            itemKey: `hsaContributionsTotal`,
          },
        ]}
      />
      <SummaryTable
        i18nKey='/info/income/hsa/hsa-contributions-summary-under'
        batches={[`hsa-0`]}
        conditions={[
          { operator: `isTrue`, condition: `/secondaryFiler/hasHsaContributionsNotExceedLimits` },
          { operator: `isTrue`, condition: `/isFilingStatusMFJ` },
          { operator: `isTrueOrIncomplete`, condition: `/secondaryFiler/secondaryFilerHasMadeContributionsToHsa` },
        ]}
        items={[
          {
            itemKey: `hsaContributionsSpouse`,
          },
          {
            itemKey: `hsaContributionsTotalSpouse`,
          },
        ]}
      />
      <InfoDisplay i18nKey='/info/income/hsa/hsa-contributions-summary-under-info' batches={[`hsa-0`]} />
      <SaveAndOrContinueButton />
    </Screen>

    <Gate condition='/someFilerCanHaveDistribution'>
      {/* Distributions section */}
      {distributionsSubSubCategory}
    </Gate>
  </Subcategory>
);
