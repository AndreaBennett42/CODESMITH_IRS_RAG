/* eslint-disable max-len */
import { Gate, Screen, SubSubcategory, Subcategory } from '../../flowDeclarations.js';
import {
  ContextHeading,
  Enum,
  TaxReturnAlert,
  Heading,
  IconDisplay,
  InfoDisplay,
  InternalLink,
  IntroContent,
  SaveAndOrContinueButton,
  SetFactAction,
  ConditionalList,
  CollectionItemReference,
  CollectionDataViewInternalLink,
  DFAccordion,
  DFModal,
  DFAlert,
  KnockoutButton,
} from '../../ContentDeclarations.js';

export const FilingStatusSubcategory = (
  <Subcategory
    route='filing-status'
    completeIf='/isFilingStatusComplete'
    dataItems={[
      {
        itemKey: `filingStatus`,
      },
    ]}
  >
    <SubSubcategory route='filing-status' hidden={true}>
      {/* This section is different based on if the user has completed it or not, since most of its screens
        are assertions (or allowing overrides of assertions, as opposed to collecting data.*/}
      <Gate condition={{ operator: `isIncomplete`, condition: `/filingStatus` }}>
        {/* This section of the filing status flow is for people who haven't yet elected that they want a joint return
        or a separate return. People who, in the spouse section, lived apart for 6 months and wanted to qualify for HoH
        _will_ enter this section, which will likely encourage them to file MFJ (and then let them redecide to MFS)
     */}
        <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/tpSelectedStatusInSpouseSectionAndCanUseIt` }}>
          <Screen route='filing-status-intro'>
            <ContextHeading displayOnlyOn='edit' i18nKey='/heading/you-and-your-family/filing-status/filing-status' />
            <Heading i18nKey='/heading/you-and-your-family/filing-status/filing-status-intro' />
            <IntroContent i18nKey='/info/you-and-your-family/filing-status/intro' />
            <SaveAndOrContinueButton />
          </Screen>

          <Gate condition='/showSingleFilingStatusScreen'>
            <Screen route='filing-status-assertion-single-only'>
              <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/single' />
              <InfoDisplay
                i18nKey='/info/you-and-your-family/filing-status/single-never-married'
                condition={{ operator: `isFalse`, condition: `/isWidowed` }}
              />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/single-widowed' condition='/isWidowed' />
              <DFAccordion
                i18nKey='/info/you-and-your-family/filing-status/filing-status-reqs-single'
                headingLevel='h2'
              />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
              <DFModal i18nKey='/info/you-and-your-family/filing-status/reasons-help' />
              <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>

          <Gate condition='/showMFSFilingStatusScreen'>
            <Screen route='filing-status-assertion-mfs-only'>
              <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/married-filing-separately' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/married-filing-separately' />
              <DFAccordion i18nKey='/info/you-and-your-family/filing-status/filing-status-reqs-mfs' headingLevel='h2' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
              <DFModal i18nKey='/info/you-and-your-family/filing-status/reasons-help' />
              <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>

          <Gate condition='/showHoHFilingStatusScreen'>
            <Screen route='filing-status-assertion-hoh-best'>
              <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/head-of-household' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/head-of-household' />
              <ConditionalList
                i18nKey='/info/you-and-your-family/filing-status/head-of-household-list'
                items={[
                  // eligibleForHoh = /dependents/*/hohQualifyingPerson" > 0
                  {
                    // HOH Test A1: IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo' (**Note: This test (HOH Test A1) must always be paired with HOH Test B1)
                    itemKey: `listItem_hoh_lived_apart_unmarried`,
                    conditions: [
                      `/marriedAndLivedApartSecondaryFilerResidencyEligible`, // if you lived apart from a noncitizen spouse you should not see this
                    ],
                  },
                  {
                    // HOH Test A2: IF qualified bc marital status was never married (single), divorced/separated, or widowed
                    itemKey: `listItem_hoh_not_married`,
                    conditions: [{ operator: `isFalse`, condition: `/maritalStatusAllowsFilingMarried` }],
                  },
                  {
                    // HOH Test A3: IF qualified bc met tests to be considered unmarried via spouse citizenship
                    itemKey: `listItem_hoh_spouse_noncitizen`,
                    // Sourced from /NRASpouseTryingForHoh, this fact wraps any: /SF/isUsCitizenFullYear, citizenAtEndOfTaxYear isNoncitizenResidentFullYear
                    conditions: [
                      `/isMarried`,
                      { operator: `isFalse`, condition: `/secondaryFilerResidencyEligibleForMFJ` },
                    ],
                  },
                  {
                    //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                    itemKey: `listItem_hoh_child_livedapart`,
                    conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
                  },
                  {
                    //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                    itemKey: `listItem_hoh_child_livedapart_pt2`,
                    conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
                  },
                  {
                    // HOH Test B2: QP IF relationship == bio/adoptive father or mother
                    itemKey: `listItem_hoh_qualified_parent_mf`,
                    conditions: [`/hasHohQualifyingParent`],
                  },
                  {
                    // HOH Test B3: QP IF relationship != bio/adoptive father or mother
                    // B3 should cover all relationships that are not "parent/mother or father" where the taxpayer was really not married or was unmarried because their spouse was nonresident
                    // lived apart
                    itemKey: `listItem_hoh_not_motherfather`,
                    conditions: [
                      { operator: `isFalse`, condition: `/hasHohQualifyingParent` },
                      { operator: `isFalse`, condition: `/marriedAndLivedApartSecondaryFilerResidencyEligible` },
                    ],
                  },
                ]}
              />
              <DFAlert
                i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit'
                type='info'
                headingLevel='h2'
              >
                <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-modal' />
              </DFAlert>
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
              <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
              <SaveAndOrContinueButton />
              <InternalLink
                condition='/hasMultipleEligibleFilingStatuses'
                i18nKey='/info/filing-status/choose-another'
                route='/flow/you-and-your-family/filing-status/filing-status-override'
              />
            </Screen>
          </Gate>

          <Gate condition='/showQSSFilingStatusScreen'>
            <Screen route='filing-status-assertion-qss-best'>
              <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/qualified-surviving-spouse' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/qualified-surviving-preface' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/qualified-surviving-spouse' />
              <DFAlert
                i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit'
                type='info'
                headingLevel='h2'
              >
                <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-modal' />
              </DFAlert>
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
              <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
              <SaveAndOrContinueButton />
              <InternalLink
                condition='/hasMultipleEligibleFilingStatuses'
                i18nKey='/info/filing-status/choose-another'
                route='/flow/you-and-your-family/filing-status/filing-status-override'
              />
            </Screen>
          </Gate>

          <Gate condition='/showMFJFilingStatusScreen'>
            <Screen route='filing-status-assertion-mfj-best'>
              <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/married-filing-jointly' />
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly' />
              <ConditionalList
                i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
                items={[
                  {
                    itemKey: `listItem_mfj_married`,
                    conditions: [`/isMarried`],
                  },
                  {
                    itemKey: `listItem_mfj_widowed_in_tax_year`,
                    conditions: [`/isWidowedInTaxYear`],
                  },
                  {
                    itemKey: `listItem_mfj_citizen_status`,
                  },
                ]}
              />
              <DFAlert
                i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj-hoh'
                type='info'
                headingLevel='h2'
              >
                <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj-hoh-modal' />
              </DFAlert>
              <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
              <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
              <SaveAndOrContinueButton />
              <InternalLink
                condition='/hasMultipleEligibleFilingStatuses'
                i18nKey='/info/filing-status/choose-another'
                route='/flow/you-and-your-family/filing-status/filing-status-override'
              />
            </Screen>
          </Gate>
        </Gate>

        <Gate condition='/tpSelectedStatusInSpouseSectionAndCanUseIt'>
          {/* If we're going through the first time, we need to do manual choice to set the filing status fact */}
          <Screen route='filing-status-manual-choice-first-time'>
            <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
            <Heading
              i18nKey='/heading/you-and-your-family/filing-status/manual-choice-first-time-mfs'
              condition='/treatAsMFS'
            />
            <Heading
              i18nKey='/heading/you-and-your-family/filing-status/manual-choice-first-time-mfj'
              condition='/treatAsMFJ'
            />
            {/* MFS */}
            <InfoDisplay
              condition='/treatAsMFS'
              i18nKey='/info/you-and-your-family/filing-status/married-filing-separately'
            />
            {/* MFJ */}
            <InfoDisplay
              i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
              condition='/treatAsMFJ'
            />
            <ConditionalList
              condition='/treatAsMFJ'
              i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
              items={[
                {
                  itemKey: `listItem_mfj_married`,
                  conditions: [`/isMarried`],
                },
                {
                  itemKey: `listItem_mfj_widowed_in_tax_year`,
                  conditions: [`/isWidowedInTaxYear`],
                },
                {
                  itemKey: `listItem_mfj_citizen_status`,
                },
              ]}
            />
            <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
            <DFModal i18nKey='/info/you-and-your-family/filing-status/reasons-help' />
            <SetFactAction path='/filingStatus' source='/statusSelectedInSpouseSection' />
            <SaveAndOrContinueButton />
            <InternalLink
              condition='/hasMultipleEligibleFilingStatuses'
              i18nKey='/info/filing-status/choose-another'
              route='/flow/you-and-your-family/filing-status/filing-status-override'
            />
          </Screen>
        </Gate>
      </Gate>
      <Gate condition={{ operator: `isComplete`, condition: `/filingStatus` }}>
        {/* You might be wondering, why is the override before any of the actual status screens? Well,
        it isn't routed automatically, and statuses that include it all link to it manually, so by putting it
        first, we get a free looping behavior that the mocks specify without any additional components */}
        <Screen route='filing-status-override' routeAutomatically={false}>
          <Heading i18nKey='/heading/you-and-your-family/filing-status/filing-status-override' />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/filing-status-override' />
          <DFAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit'
            type='info'
            headingLevel='h2'
            condition={{ operator: `isFalse`, condition: `/eligibleForMFJ` }} // if some combo of QSS, Single, MFS, HoH only (!MFJ) use generic alert template
          >
            <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-modal' />
          </DFAlert>
          <DFAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj'
            type='info'
            headingLevel='h2'
            conditions={[`/eligibleForMFJ`, { operator: `isFalse`, condition: `/eligibleForHoh` }]} // if MFJ only, show shorter mfj alert
          >
            <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj-modal' />
          </DFAlert>
          <DFAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj-hoh'
            type='info'
            headingLevel='h2'
            conditions={[`/eligibleForMFJ`, `/eligibleForHoh`]} // if MFJ or HoH show longer mfj alert
          >
            <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-mfj-hoh-modal' />
          </DFAlert>
          <Enum path='/filingStatus' i18nKeySuffixContext='long' />
          <SaveAndOrContinueButton
            condition='/hasMultipleEligibleFilingStatuses'
            nextRouteOverride='/flow/you-and-your-family/filing-status/filing-status-manual-choice'
          />
          <SaveAndOrContinueButton
            condition={{ operator: `isFalse`, condition: `/hasMultipleEligibleFilingStatuses` }}
            nextRouteOverride='/flow/you-and-your-family/filing-status/filing-status-no-choice'
          />
        </Screen>

        <Screen
          route='filing-status-manual-choice'
          condition='/hasValidFillingStatusWithChoice'
          routeAutomatically={false}
          actAsDataView={true}
        >
          <DFAlert
            i18nKey='/info/hoh-qp-error-generic'
            condition='/needsToRevisitHohQualifyingPersonSelectionOrFilingStatus'
            type='error'
            headingLevel='h2'
          />

          <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
          <Heading i18nKey='/heading/you-and-your-family/filing-status/manual-choice' />

          {/* Intro text, dont show for single or mfs since they are only one sentence */}
          <InfoDisplay
            i18nKey='/info/you-and-your-family/filing-status/manual-choice'
            conditions={[
              { operator: `isFalse`, condition: `/isFilingStatusSingle` },
              { operator: `isFalse`, condition: `/isFilingStatusMFS` },
            ]}
          />

          {/* Single */}
          <InfoDisplay
            conditions={[
              `/isFilingStatusSingle`,
              { operator: `isFalse`, condition: `/isMarried` },
              { operator: `isFalse`, condition: `/isWidowed` },
            ]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />

          <InfoDisplay
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, `/eligibleForQss`]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />

          <InfoDisplay
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, { operator: `isFalse`, condition: `/eligibleForQss` }]}
            i18nKey='/info/you-and-your-family/filing-status/single-widowed'
          />

          {/* QSS */}
          <InfoDisplay
            condition='/isFilingStatusQSS'
            i18nKey='/info/you-and-your-family/filing-status/qualified-surviving-spouse'
          />

          {/* MFJ */}
          <ConditionalList
            condition='/isFilingStatusMFJ'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
            items={[
              {
                itemKey: `listItem_mfj_married`,
                conditions: [`/isMarried`],
              },
              {
                itemKey: `listItem_mfj_widowed_in_tax_year`,
                conditions: [`/isWidowedInTaxYear`],
              },
              {
                itemKey: `listItem_mfj_citizen_status`,
              },
            ]}
          />

          {/* MFS */}
          <InfoDisplay
            condition='/isFilingStatusMFS'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-separately'
          />

          {/* HOH */}
          <ConditionalList
            condition='/isFilingStatusHOH'
            i18nKey='/info/you-and-your-family/filing-status/head-of-household-list'
            items={[
              // eligibleForHoh = /dependents/*/hohQualifyingPerson" > 0
              {
                // HOH Test A1: IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo' (**Note: This test (HOH Test A1) must always be paired with HOH Test B1)
                itemKey: `listItem_hoh_lived_apart_unmarried`,
                conditions: [
                  `/marriedAndLivedApartSecondaryFilerResidencyEligible`, // if you lived apart from a noncitizen spouse you should not see this
                ],
              },
              {
                // HOH Test A2: IF qualified bc marital status was never married (single), divorced/separated, or widowed
                itemKey: `listItem_hoh_not_married`,
                conditions: [{ operator: `isFalse`, condition: `/maritalStatusAllowsFilingMarried` }],
              },
              {
                // HOH Test A3: IF qualified bc met tests to be considered unmarried via spouse citizenship
                itemKey: `listItem_hoh_spouse_noncitizen`,
                // Sourced from /NRASpouseTryingForHoh, this fact wraps any: /SF/isUsCitizenFullYear, citizenAtEndOfTaxYear isNoncitizenResidentFullYear
                conditions: [
                  `/isMarried`,
                  { operator: `isFalse`, condition: `/secondaryFilerResidencyEligibleForMFJ` },
                ],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart_pt2`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                // HOH Test B2: QP IF relationship == bio/adoptive father or mother
                itemKey: `listItem_hoh_qualified_parent_mf`,
                conditions: [`/hasHohQualifyingParent`],
              },
              {
                // HOH Test B3: QP IF relationship != bio/adoptive father or mother
                // B3 should cover all relationships that are not "parent/mother or father" where the taxpayer was really not married or was unmarried because their spouse was nonresident
                // lived apart
                itemKey: `listItem_hoh_not_motherfather`,
                conditions: [
                  { operator: `isFalse`, condition: `/hasHohQualifyingParent` },
                  { operator: `isFalse`, condition: `/marriedAndLivedApartSecondaryFilerResidencyEligible` },
                ],
              },
            ]}
          />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/manual-choice-closing' />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
          <DFModal i18nKey='/info/you-and-your-family/filing-status/reasons-help' />
          <SaveAndOrContinueButton />
          <InternalLink
            condition='/hasMultipleEligibleFilingStatuses'
            i18nKey='/info/filing-status/choose-another'
            route='/flow/you-and-your-family/filing-status/filing-status-override'
          />
        </Screen>

        <Screen
          route='community-property-ko-go-back'
          condition='/knockoutSubjectToCommunityPropertyLaws'
          isKnockout={true}
        >
          <IconDisplay name='ErrorOutline' size={9} isCentered />
          <Heading i18nKey='/heading/knockout/forms-missing/generic-doesnt-have-tax-forms' />
          <InfoDisplay i18nKey='/info/knockout/forms-missing/community-property-ko' />
          <DFAlert i18nKey='/info/knockout/community-property-ko-go-back' headingLevel='h2' type='warning' />
          <KnockoutButton i18nKey='button.knockout' />
        </Screen>

        <Screen
          route='filing-status-no-choice'
          condition='/hasValidFilingStatusWithNoChoice'
          actAsDataView={true}
          routeAutomatically={false}
        >
          <IconDisplay name='InfoOutline' size={9} isCentered className='text-primary' />
          <Heading i18nKey='/heading/you-and-your-family/filing-status/no-choice' />
          {/* Intro text, dont show for single or mfs since they are only one sentence */}
          <InfoDisplay
            i18nKey='/info/you-and-your-family/filing-status/manual-choice'
            conditions={[
              { operator: `isFalse`, condition: `/isFilingStatusSingle` },
              { operator: `isFalse`, condition: `/isFilingStatusMFS` },
            ]}
          />
          {/* Single */}
          <InfoDisplay
            conditions={[
              `/isFilingStatusSingle`,
              { operator: `isFalse`, condition: `/isMarried` },
              { operator: `isFalse`, condition: `/isWidowed` },
            ]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />
          {/* Repeat this instead of making a new fact. There may be a more clever way to do this */}
          <InfoDisplay
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, `/eligibleForQss`]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />
          <InfoDisplay
            i18nKey='/info/you-and-your-family/filing-status/single-widowed'
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, { operator: `isFalse`, condition: `/eligibleForQss` }]}
          />
          {/* QSS */}
          <InfoDisplay
            condition='/isFilingStatusQSS'
            i18nKey='/info/you-and-your-family/filing-status/qualified-surviving-spouse'
          />
          {/* MFJ */}
          <ConditionalList
            condition='/isFilingStatusMFJ'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
            items={[
              {
                itemKey: `listItem_mfj_married`,
                conditions: [`/isMarried`],
              },
              {
                itemKey: `listItem_mfj_widowed_in_tax_year`,
                conditions: [`/isWidowedInTaxYear`],
              },
              {
                itemKey: `listItem_mfj_citizen_status`,
              },
            ]}
          />
          {/* MFS */}
          <InfoDisplay
            condition='/isFilingStatusMFS'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-separately'
          />
          {/* HOH */}
          <ConditionalList
            condition='/isFilingStatusHOH'
            i18nKey='/info/you-and-your-family/filing-status/head-of-household-list'
            items={[
              // eligibleForHoh = /dependents/*/hohQualifyingPerson" > 0
              {
                // HOH Test A1: IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo' (**Note: This test (HOH Test A1) must always be paired with HOH Test B1)
                itemKey: `listItem_hoh_lived_apart_unmarried`,
                conditions: [
                  `/marriedAndLivedApartSecondaryFilerResidencyEligible`, // if you lived apart from a noncitizen spouse you should not see this
                ],
              },
              {
                // HOH Test A2: IF qualified bc marital status was never married (single), divorced/separated, or widowed
                itemKey: `listItem_hoh_not_married`,
                conditions: [{ operator: `isFalse`, condition: `/maritalStatusAllowsFilingMarried` }],
              },
              {
                // HOH Test A3: IF qualified bc met tests to be considered unmarried via spouse citizenship
                itemKey: `listItem_hoh_spouse_noncitizen`,
                // Sourced from /NRASpouseTryingForHoh, this fact wraps any: /SF/isUsCitizenFullYear, citizenAtEndOfTaxYear isNoncitizenResidentFullYear
                conditions: [
                  `/isMarried`,
                  { operator: `isFalse`, condition: `/secondaryFilerResidencyEligibleForMFJ` },
                ],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart_pt2`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                // HOH Test B2: QP IF relationship == bio/adoptive father or mother
                itemKey: `listItem_hoh_qualified_parent_mf`,
                conditions: [`/hasHohQualifyingParent`],
              },
              {
                // HOH Test B3: QP IF relationship != bio/adoptive father or mother
                // B3 should cover all relationships that are not "parent/mother or father" where the taxpayer was really not married or was unmarried because their spouse was nonresident
                // lived apart
                itemKey: `listItem_hoh_not_motherfather`,
                conditions: [
                  { operator: `isFalse`, condition: `/hasHohQualifyingParent` },
                  { operator: `isFalse`, condition: `/marriedAndLivedApartSecondaryFilerResidencyEligible` },
                ],
              },
            ]}
          />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/updating-disclaimer' />
          <DFModal i18nKey='/info/you-and-your-family/filing-status/reasons-help' />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen
          route='filing-status-error-autocorrect'
          condition='/hasInvalidFilingStatusWithNoChoice'
          actAsDataView={true}
        >
          <TaxReturnAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-error-autocorrect'
            condition='/hasInvalidFilingStatusWithNoChoice'
            type='error'
          />
          <Heading i18nKey='/heading/you-and-your-family/filing-status/filing-status-error-autocorrect' />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/filing-status-error-autocorrect' />

          {/* Intro text, dont show for single or mfs since they are only one sentence */}
          <InfoDisplay
            i18nKey='/info/you-and-your-family/filing-status/manual-choice'
            conditions={[
              { operator: `isFalse`, condition: `/isFilingStatusSingle` },
              { operator: `isFalse`, condition: `/isFilingStatusMFS` },
            ]}
          />

          {/* Single */}
          <InfoDisplay
            conditions={[
              `/isFilingStatusSingle`,
              { operator: `isFalse`, condition: `/isMarried` },
              { operator: `isFalse`, condition: `/isWidowed` },
            ]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />

          <InfoDisplay
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, `/eligibleForQss`]}
            i18nKey='/info/you-and-your-family/filing-status/single-never-married'
          />

          <InfoDisplay
            conditions={[`/isFilingStatusSingle`, `/isWidowed`, { operator: `isFalse`, condition: `/eligibleForQss` }]}
            i18nKey='/info/you-and-your-family/filing-status/single-widowed'
          />

          {/* QSS */}
          <InfoDisplay
            condition='/isFilingStatusQSS'
            i18nKey='/info/you-and-your-family/filing-status/qualified-surviving-spouse'
          />

          {/* MFJ */}
          <ConditionalList
            condition='/isFilingStatusMFJ'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-jointly'
            items={[
              {
                itemKey: `listItem_mfj_married`,
                conditions: [`/isMarried`],
              },
              {
                itemKey: `listItem_mfj_widowed_in_tax_year`,
                conditions: [`/isWidowedInTaxYear`],
              },
              {
                itemKey: `listItem_mfj_citizen_status`,
              },
            ]}
          />

          {/* MFS */}
          <InfoDisplay
            condition='/isFilingStatusMFS'
            i18nKey='/info/you-and-your-family/filing-status/married-filing-separately'
          />

          {/* HOH */}
          <ConditionalList
            condition='/isFilingStatusHOH'
            i18nKey='/info/you-and-your-family/filing-status/head-of-household-list'
            items={[
              // eligibleForHoh = /dependents/*/hohQualifyingPerson" > 0
              {
                // HOH Test A1: IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo' (**Note: This test (HOH Test A1) must always be paired with HOH Test B1)
                itemKey: `listItem_hoh_lived_apart_unmarried`,
                conditions: [
                  `/marriedAndLivedApartSecondaryFilerResidencyEligible`, // if you lived apart from a noncitizen spouse you should not see this
                ],
              },
              {
                // HOH Test A2: IF qualified bc marital status was never married (single), divorced/separated, or widowed
                itemKey: `listItem_hoh_not_married`,
                conditions: [{ operator: `isFalse`, condition: `/maritalStatusAllowsFilingMarried` }],
              },
              {
                // HOH Test A3: IF qualified bc met tests to be considered unmarried via spouse citizenship
                itemKey: `listItem_hoh_spouse_noncitizen`,
                // Sourced from /NRASpouseTryingForHoh, this fact wraps any: /SF/isUsCitizenFullYear, citizenAtEndOfTaxYear isNoncitizenResidentFullYear
                conditions: [
                  `/isMarried`,
                  { operator: `isFalse`, condition: `/secondaryFilerResidencyEligibleForMFJ` },
                ],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                //HOH Test B1:QP IF relationship == bio/adopted child, stepchild, foster child AND IF qualified bc met tests to be considered unmarried via 'lived apart last 6mo'QP IF relationship != bio/adoptive father or mother
                itemKey: `listItem_hoh_child_livedapart_pt2`,
                conditions: [`/marriedAndLivedApartSecondaryFilerResidencyEligible`, `/hasHohQualifyingChild`],
              },
              {
                // HOH Test B2: QP IF relationship == bio/adoptive father or mother
                itemKey: `listItem_hoh_qualified_parent_mf`,
                conditions: [`/hasHohQualifyingParent`],
              },
              {
                // HOH Test B3: QP IF relationship != bio/adoptive father or mother
                // B3 should cover all relationships that are not "parent/mother or father" where the taxpayer was really not married or was unmarried because their spouse was nonresident
                // lived apart
                itemKey: `listItem_hoh_not_motherfather`,
                conditions: [
                  { operator: `isFalse`, condition: `/hasHohQualifyingParent` },
                  { operator: `isFalse`, condition: `/marriedAndLivedApartSecondaryFilerResidencyEligible` },
                ],
              },
            ]}
          />
          <SetFactAction path='/filingStatus' source='/recommendedFilingStatus' />
          <SaveAndOrContinueButton />
        </Screen>

        <Screen
          route='filing-status-error-manual-fix'
          condition='/hasInvalidFilingStatusWithChoice'
          actAsDataView={true}
        >
          <TaxReturnAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-error-manual-fix'
            condition='/hasInvalidFilingStatusWithChoice'
            type='error'
          />
          <Heading i18nKey='/heading/you-and-your-family/filing-status/filing-status-error-manual-fix' />
          <InfoDisplay i18nKey='/info/you-and-your-family/filing-status/filing-status-error-manual-fix' />
          <DFAlert
            i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit'
            type='info'
            headingLevel='h2'
          >
            <DFModal i18nKey='/info/you-and-your-family/filing-status/filing-status-best-fit-modal' />
          </DFAlert>
          <Enum path='/filingStatus' i18nKeySuffixContext='capitalized' />
          <SaveAndOrContinueButton />
        </Screen>

        {/* The user chose MFJ, but more MFJ info must be input. Eventually this has to become a modal? */}
        <Screen route='more-spouse-info-a' condition='/needsMoreMFJSpouseInfo'>
          <Heading i18nKey='/heading/you-and-your-family/filing-status/more-spouse-info-dialog' />
          <InternalLink
            i18nKey='/info/filing-status/go-to-spouse'
            route='/flow/you-and-your-family/spouse/add-spouse-a'
            displayAsButton={true}
          />
        </Screen>
        {/* The user chose MFS, but more MFS info must be input. Eventually this has to become a modal? */}
        <Screen route='more-spouse-info-b' condition='/needsMoreMFSSpouseInfo'>
          <Heading i18nKey='/heading/you-and-your-family/filing-status/more-spouse-info-dialog' />
          <InternalLink
            i18nKey='/info/filing-status/go-to-spouse'
            route='/flow/you-and-your-family/spouse/add-spouse-b'
            displayAsButton={true}
          />
        </Screen>

        {/* The user chose HoH, and they need to select a QP */}
        <Gate condition='/isFilingStatusHOH'>
          {/* If the user switched from MFJ to HoH, we will have cleared their people's relationship statuses
            so we need to send them back to the family and household section */}
          <Gate condition={{ operator: `isFalseOrIncomplete`, condition: `/eligibleForHoh` }}>
            <Screen route='incomplete-family-information'>
              <Heading i18nKey='/heading/you-and-your-family/filing-status/more-family-info' />
              <InternalLink
                i18nKey='/info/filing-status/go-to-family-hh'
                route='/data-view/flow/you-and-your-family/dependents'
                displayAsButton={true}
              />
            </Screen>
          </Gate>
          <Gate condition='/hasExactlyOneHohQP'>
            <Screen route='hoh-qp-name'>
              <TaxReturnAlert
                i18nKey={`/info/hoh-qp-error`}
                conditions={[`/needsToRevisitHohQualifyingPersonSelectionOrFilingStatus`, `/hasExactlyOneHohQP`]}
                type='error'
              />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/hoh-qp-name' />
              <InfoDisplay
                i18nKey='/info/you-and-your-family/filing-status/hoh-qp-qp-name'
                condition={{ operator: `isFalse`, condition: `/firstHohQP/isClaimedDependent` }}
              />
              <SetFactAction source='/firstHohQP' path='/hohQualifyingPerson' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Gate condition={{ operator: `isFalse`, condition: `/hasExactlyOneHohQP` }}>
            <Screen route='hoh-qp-selection'>
              <TaxReturnAlert
                i18nKey={`/info/hoh-qp-error-select`}
                conditions={[
                  `/needsToRevisitHohQualifyingPersonSelectionOrFilingStatus`,
                  { operator: `isFalse`, condition: `/hasExactlyOneHohQP` },
                ]}
                type='error'
              />
              <Heading i18nKey='/heading/you-and-your-family/filing-status/hoh-qp-selection' />
              <InfoDisplay i18nKey='/info/you-and-you-family/filing-status/hoh-qp-selection' />
              <InfoDisplay
                i18nKey='/info/you-and-you-family/filing-status/hoh-qp-selection-tin'
                condition={{ operator: `isFalse`, condition: `/allHoHQPsAreClaimedDependents` }}
              />
              <CollectionItemReference path='/hohQualifyingPerson' />
              <SaveAndOrContinueButton />
            </Screen>
          </Gate>
          <Screen route='hoh-qp-no-tin' condition='/knockoutHohQpHasNoTin' isKnockout={true}>
            <IconDisplay name='ErrorOutline' size={9} isCentered />
            <Heading i18nKey='/heading/knockout/hoh-qp-no-tin' />
            <InfoDisplay i18nKey='/info/knockout/hoh-qp-no-tin' />
            <DFAlert i18nKey='/info/knockout/how-to-file-paper' headingLevel='h2' type='warning' />
            <KnockoutButton i18nKey='button.knockout' />
          </Screen>
          <Screen route='more-hoh-qp-info-dialog' condition='/needToEnterHohQPTin'>
            <Heading i18nKey='/heading/you-and-your-family/filing-status/more-hoh-qp-info-dialog' />
            <CollectionDataViewInternalLink
              collectionItemPath='/hohQualifyingPerson'
              i18nKey='/info/filing-status/go-to-family-hh'
              dataViewUrl={`/data-view/loop/${encodeURIComponent(`/familyAndHousehold`)}`}
            />
          </Screen>
        </Gate>
      </Gate>
    </SubSubcategory>
  </Subcategory>
);
