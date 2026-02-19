import { Gate, Screen, SubSubcategory, CollectionLoop } from '../../flowDeclarations.js';
import {
  Address,
  Boolean,
  CollectionItemManager,
  CollectionItemReference,
  Ein,
  LimitingString,
  Heading,
  InfoDisplay,
  SaveAndOrContinueButton,
  Tin,
  Dollar,
  ConditionalList,
  Hint,
  DFModal,
  MultiEnum,
  InternalLink,
  ContextHeading,
  TaxReturnAlert,
} from '../../ContentDeclarations.js';
import { RawCondition } from '../../Condition.js';

type DisplayState = 'display-on' | 'display-off';
export const makeCdccCareProvidersCollection = (
  loopName: string,
  showContextHeading: boolean,
  condition: RawCondition
) => {
  const contextHeadingSuffix: DisplayState = showContextHeading ? `display-on` : `display-off`;
  return (
    <Gate condition={condition}>
      <SubSubcategory route='care-providers' completeIf='/cdccSectionIsComplete' collectionContext='/cdccCareProviders'>
        <Screen route='provider-collection-hub' hasScreenRouteOverride>
          <ContextHeading
            i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
            batches={[`cdcc-0`]}
          />
          <TaxReturnAlert
            type='error'
            i18nKey='/info/cdcc-shared/care-providers/at-least-one'
            batches={[`cdcc-2`]}
            conditions={[
              { operator: `isFalseOrIncomplete`, condition: `/hasAtLeastOneCareProvider` },
              `/cdccCareProvidersIsDone`,
              // TODO: Alert bubbles up to the checklist if we don't duplciate the top-level condition here
              condition,
            ]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/care-providers/cdcc-care-providers-hub-intro'
            conditions={[
              { operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` },
              { operator: `isFalse`, condition: `/hasAtLeastOneCareProvider` },
            ]}
            batches={[`cdcc-3`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/care-providers/cdcc-care-providers-hub-people-intro'
            conditions={[
              `/cdccMoreThanOneQualifyingPerson`,
              { operator: `isFalse`, condition: `/hasAtLeastOneCareProvider` },
            ]}
            batches={[`cdcc-3`]}
          />
          <Heading
            i18nKey='/heading/credits-and-deductions/care-providers/cdcc-care-providers-hub'
            condition='/hasAtLeastOneCareProvider'
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/care-providers/cdcc-care-providers-hub-intro'
            conditions={[
              { operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` },
              `/hasAtLeastOneCareProvider`,
            ]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/care-providers/cdcc-care-providers-hub-intro-people'
            conditions={[`/cdccMoreThanOneQualifyingPerson`, `/hasAtLeastOneCareProvider`]}
          />
          <ConditionalList
            i18nKey='/heading/credits-and-deductions/care-providers/qualifying-person'
            batches={[`cdcc-3`]}
            items={[
              { itemKey: `qps`, collection: `/cdccQualifyingNonFilersWithExpenses` },
              { itemKey: `qfilers`, collection: `/cdccQualifyingFilersWithExpenses` },
            ]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/care-providers/cdcc-care-providers-hub'
            condition='/hasCdccImmediateChildQpUnder13'
            batches={[`cdcc-3`]}
          />
          <InfoDisplay
            i18nKey='/info/credits-and-deductions/care-providers/cdcc-care-providers-hub-sans-parent-bullet'
            condition={{ operator: `isFalse`, condition: `/hasCdccImmediateChildQpUnder13` }}
            batches={[`cdcc-3`]}
          />
          <CollectionItemManager
            path='/cdccCareProviders'
            loopName={loopName}
            donePath='/cdccCareProvidersIsDone'
            batches={[`cdcc-2`]}
          />
        </Screen>
        <CollectionLoop
          loopName={loopName}
          collection={`/cdccCareProviders`}
          collectionItemCompletedCondition='/cdccCareProviders/*/isComplete'
          hideCardLabel2Condition='/cdccCareProviders/*/isEmployerFurnishedWithW2'
          donePath='/cdccCareProvidersIsDone'
          isInner={true}
          shouldSeeHubCompletionBtnsPath='/hasAtLeastOneCareProvider'
          dataReveals={[
            {
              itemKey: `num-care-providers-one`,
              conditions: [{ operator: `isTrue`, condition: `/hasExactlyOneCompleteCareProvider` }],
            },
            {
              itemKey: `num-care-providers-zero-many`,
              conditions: [{ operator: `isFalse`, condition: `/hasExactlyOneCompleteCareProvider` }],
            },
          ]}
        >
          <SubSubcategory route='care-providers-info'>
            <Screen route='provider-bus-or-indiv'>
              <ContextHeading
                displayOnlyOn='edit'
                i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                batches={[`cdcc-0`]}
              />
              <Heading
                i18nKey='/heading/credits-and-deductions/care-providers/provider-bus-or-indiv'
                batches={[`cdcc-0`]}
              />
              <InfoDisplay
                i18nKey='/info/credits-and-deductions/care-providers/provider-bus-or-indiv'
                batches={[`cdcc-0`]}
              />
              <Boolean path='/cdccCareProviders/*/writableIsOrganization' batches={[`cdcc-3`]} />
              <SaveAndOrContinueButton />
            </Screen>
            <Gate condition='/cdccQualifiedForBenefit'>
              <Screen route='provider-employer' condition='/cdccQualifiedForBenefit'>
                <ContextHeading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                  batches={[`cdcc-0`]}
                />
                <Heading
                  condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
                  i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-q'
                  batches={[`cdcc-0`]}
                />
                <Heading
                  condition='/isFilingStatusMFJ'
                  i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-q-mfj'
                  batches={[`cdcc-0`]}
                />
                <InfoDisplay
                  condition={{ operator: `isFalse`, condition: `/isFilingStatusMFJ` }}
                  i18nKey='/info/credits-and-deductions/care-providers/provider-employer-q'
                  batches={[`cdcc-0`]}
                />
                <InfoDisplay
                  condition='/isFilingStatusMFJ'
                  i18nKey='/info/credits-and-deductions/care-providers/provider-employer-q-mfj'
                  batches={[`cdcc-0`]}
                />
                <Boolean path='/cdccCareProviders/*/isEmployerFurnished' batches={[`cdcc-3`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <Gate condition='/cdccCareProviders/*/isEmployerFurnished'>
                <Gate condition={{ operator: `isFalse`, condition: `/filersHaveExactlyOneW2` }}>
                  <Screen route='provider-employer-check-multiple'>
                    <ContextHeading
                      i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-check-multiple'
                      batches={[`cdcc-0`]}
                    />
                    <InfoDisplay
                      i18nKey='/info/credits-and-deductions/care-providers/provider-employer-check-multiple'
                      batches={[`cdcc-0`]}
                    />
                    <ConditionalList
                      i18nKey='/info/credits-and-deductions/care-providers/form-w2-employers'
                      items={[{ itemKey: `primary`, collection: `/formW2s` }]}
                      batches={[`cdcc-2`]}
                    />
                    <Boolean path='/cdccCareProviders/*/hasW2Employer' displayOnlyOn='edit' batches={[`cdcc-2`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen route='provider-employer-selection' condition='/cdccCareProviders/*/hasW2Employer'>
                    <ContextHeading
                      i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-selection'
                      batches={[`cdcc-0`]}
                    />
                    <CollectionItemReference
                      path='/cdccCareProviders/*/writableEmployerWhoFurnishedCare'
                      displayOnlyOn='edit'
                      batches={[`cdcc-2`]}
                    />
                    <SaveAndOrContinueButton />
                  </Screen>
                </Gate>
                <Screen route='provider-employer-check-single' condition='/filersHaveExactlyOneW2'>
                  <ContextHeading
                    i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                    batches={[`cdcc-0`]}
                  />
                  <Heading
                    i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-check-single'
                    batches={[`cdcc-0`]}
                  />
                  <Boolean path='/cdccCareProviders/*/hasW2Employer' displayOnlyOn='edit' batches={[`cdcc-2`]} />
                  <SaveAndOrContinueButton />
                </Screen>
                <Screen route='provider-employer-add-new' condition='/cdccCareProviders/*/flowEmployerMustBeAdded'>
                  <ContextHeading
                    i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                    batches={[`cdcc-0`]}
                  />
                  <Heading
                    condition='/filersHaveExactlyOneW2'
                    i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-add-new-one-w2'
                    batches={[`cdcc-0`]}
                  />
                  <Heading
                    condition={{ operator: `isFalse`, condition: `/filersHaveExactlyOneW2` }}
                    i18nKey='/heading/credits-and-deductions/care-providers/provider-employer-add-new-multiple-w2s'
                    batches={[`cdcc-0`]}
                  />
                  <ConditionalList
                    condition={{ operator: `isFalse`, condition: `/filersHaveExactlyOneW2` }}
                    i18nKey='/info/credits-and-deductions/care-providers/form-w2-employers'
                    items={[{ itemKey: `primary`, collection: `/formW2s` }]}
                    batches={[`cdcc-2`]}
                  />
                  <InternalLink
                    condition='/cdccHasExactlyOneQualifyingPerson'
                    route='/data-view/flow/income/jobs'
                    i18nKey='/info/credits-and-deductions/care-providers/provider-employer-add-new-one-qp'
                    batches={[`cdcc-0`]}
                  />
                  <InternalLink
                    condition={{ operator: `isFalse`, condition: `/cdccHasExactlyOneQualifyingPerson` }}
                    route='/data-view/flow/income/jobs'
                    i18nKey='/info/credits-and-deductions/care-providers/provider-employer-add-new-many-qps'
                    batches={[`cdcc-0`]}
                  />
                  <InternalLink
                    i18nKey='/info/credits-and-deductions/care-providers/go-to-jobs'
                    route='/data-view/flow/income/jobs'
                    displayAsButton={true}
                    batches={[`cdcc-2`]}
                  />
                </Screen>
              </Gate>
            </Gate>
            <Gate condition='/cdccCareProviders/*/collectProviderDetails'>
              <Screen route='provider-name'>
                <ContextHeading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                  batches={[`cdcc-0`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/care-providers/cdcc-care-provider-name'
                  batches={[`cdcc-0`]}
                />
                <Hint
                  i18nKey='/credits-and-deductions/care-providers/provider-name-hint'
                  batches={[`cdcc-0`]}
                  conditions={[
                    { condition: `/cdccCareProviders/*/isOrganization` },
                    { operator: `isFalse`, condition: `/cdccCareProviders/*/hasW2Employer` },
                    { condition: `/cdccCarryoverAmountFromPriorTaxYear` },
                  ]}
                />
                <LimitingString
                  path='/cdccCareProviders/*/writableOrganizationName'
                  condition='/cdccCareProviders/*/isOrganization'
                  batches={[`cdcc-3`]}
                />
                <LimitingString
                  path='/cdccCareProviders/*/writableFirstName'
                  condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                  batches={[`cdcc-2`]}
                />
                <LimitingString
                  path='/cdccCareProviders/*/writableLastName'
                  condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                  batches={[`cdcc-2`]}
                />
                <SaveAndOrContinueButton />
              </Screen>
              <Screen route='provider-address'>
                <ContextHeading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                  batches={[`cdcc-0`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/care-providers/cdcc-care-provider-address'
                  batches={[`cdcc-0`]}
                />
                <Hint
                  i18nKey='/credits-and-deductions/care-providers/provider-address-hint'
                  batches={[`cdcc-0`]}
                  conditions={[
                    { condition: `/cdccCareProviders/*/isOrganization` },
                    { operator: `isFalse`, condition: `/cdccCareProviders/*/hasW2Employer` },
                    { condition: `/flowTrue` },
                  ]}
                />
                <Address
                  path='/cdccCareProviders/*/writableAddress'
                  hintKey='/info/why-cant-i-change-country'
                  batches={[`cdcc-3`]}
                />
                <SaveAndOrContinueButton />
              </Screen>

              <Screen route='provider-tax-exempt' condition='/cdccCareProviders/*/isOrganization'>
                <ContextHeading
                  i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                  batches={[`cdcc-0`]}
                />
                <Heading
                  i18nKey='/heading/credits-and-deductions/care-providers/provider-tax-exempt'
                  batches={[`cdcc-0`]}
                />
                <InfoDisplay
                  i18nKey='/info/credits-and-deductions/care-providers/provider-tax-exempt'
                  batches={[`cdcc-0`]}
                />
                <Boolean path='/cdccCareProviders/*/writableIsTaxExempt' batches={[`cdcc-3`]} />
                <SaveAndOrContinueButton />
              </Screen>
              <SubSubcategory route='care-providers-tax-id'>
                <Gate condition='/cdccCareProviders/*/flowNotTaxExemptOrganization'>
                  <Screen route='tin-ein-check'>
                    <ContextHeading
                      displayOnlyOn='edit'
                      i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      condition='/cdccCareProviders/*/isOrganization'
                      i18nKey='/heading/credits-and-deductions/care-providers/tin-ein-check-ein'
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                      i18nKey='/heading/credits-and-deductions/care-providers/tin-ein-check-tin'
                      batches={[`cdcc-0`]}
                    />
                    <DFModal
                      i18nKey='/info/credits-and-deductions/care-providers/what-is-ein'
                      condition='/cdccCareProviders/*/isOrganization'
                      batches={[`cdcc-2`]}
                    />

                    <DFModal
                      i18nKey='/info/credits-and-deductions/care-providers/what-is-ssn-itin'
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                      batches={[`cdcc-2`]}
                    />
                    <Boolean path='/cdccCareProviders/*/hasTinOrEin' batches={[`cdcc-3`]} />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='tin-missing-reason'
                    condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/hasTinOrEin` }}
                  >
                    <ContextHeading
                      i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      condition='/cdccCareProviders/*/isOrganization'
                      i18nKey='/heading/credits-and-deductions/care-providers/tin-missing-reason-ein'
                      batches={[`cdcc-0`]}
                    />
                    <Heading
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                      i18nKey='/heading/credits-and-deductions/care-providers/tin-missing-reason-tin'
                      batches={[`cdcc-0`]}
                    />
                    <MultiEnum
                      path='/cdccCareProviders/*/writableDueDiligence'
                      i18nKeySuffixContext='ein'
                      condition='/cdccCareProviders/*/isOrganization'
                      batches={[`cdcc-3`]}
                    />
                    <MultiEnum
                      path='/cdccCareProviders/*/writableDueDiligence'
                      i18nKeySuffixContext='tin'
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                      batches={[`cdcc-3`]}
                    />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Screen
                    route='provider-due-diligence'
                    condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/hasTinOrEin` }}
                  >
                    <ContextHeading i18nKey='/heading/credits-and-deductions/credits/cdcc' batches={[`cdcc-0`]} />
                    <Heading
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                      i18nKey='/heading/credits-and-deductions/care-providers/provider-due-diligence-tin'
                    />
                    <Heading
                      condition={{ condition: `/cdccCareProviders/*/isOrganization` }}
                      i18nKey='/heading/credits-and-deductions/care-providers/provider-due-diligence-ein'
                    />
                    <ConditionalList
                      i18nKey='/info/credits-and-deductions/care-providers/provider-due-diligence'
                      items={[
                        { itemKey: `formW10` },
                        { itemKey: `ssn` },
                        {
                          itemKey: `dependentCare`,
                          conditions: [
                            `/cdccCareProviders/*/isEmployerFurnished`,
                            `/cdccCareProviders/*/isOrganization`,
                          ],
                        },
                        {
                          itemKey: `ssnOrItin`,
                          conditions: [
                            { operator: `isFalseOrIncomplete`, condition: `/cdccCareProviders/*/isOrganization` },
                          ],
                        },
                        { itemKey: `ein`, conditions: [`/cdccCareProviders/*/isOrganization`] },
                      ]}
                      i18nPrefixKey='/info/credits-and-deductions/care-providers/provider-due-diligence-prefix'
                    />
                    <SaveAndOrContinueButton />
                  </Screen>
                  <Gate condition='/cdccCareProviders/*/hasTinOrEin'>
                    <Screen route='provider-ein' condition='/cdccCareProviders/*/isOrganization'>
                      <ContextHeading
                        i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                        batches={[`cdcc-0`]}
                      />
                      <Heading
                        i18nKey='/heading/credits-and-deductions/care-providers/provider-ein'
                        batches={[`cdcc-0`]}
                      />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/care-providers/provider-ein'
                        batches={[`cdcc-0`]}
                      />
                      <Ein path='/cdccCareProviders/*/writableEin' isSensitive={true} />
                      <SaveAndOrContinueButton />
                    </Screen>
                    <Screen
                      route='provider-tin'
                      condition={{ operator: `isFalse`, condition: `/cdccCareProviders/*/isOrganization` }}
                    >
                      <ContextHeading
                        i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                        batches={[`cdcc-0`]}
                      />
                      <Heading
                        i18nKey='/heading/credits-and-deductions/care-providers/provider-tin'
                        batches={[`cdcc-0`]}
                      />
                      <InfoDisplay
                        i18nKey='/info/credits-and-deductions/care-providers/provider-tin'
                        batches={[`cdcc-0`]}
                      />
                      <Tin path='/cdccCareProviders/*/writableTin' isSensitive={true} batches={[`cdcc-2`]} />
                      <SaveAndOrContinueButton />
                    </Screen>
                  </Gate>
                </Gate>
              </SubSubcategory>
              <SubSubcategory route='care-providers-amount-paid'>
                <Screen route='provider-qp-expenses'>
                  <ContextHeading
                    displayOnlyOn='edit'
                    i18nKey={`/heading/credits-and-deductions/credits/cdcc-${contextHeadingSuffix}`}
                    batches={[`cdcc-0`]}
                  />
                  <Heading
                    i18nKey='/heading/credits-and-deductions/care-providers/provider-expenses'
                    condition={{ operator: `isFalse`, condition: `/cdccMoreThanOneQualifyingPerson` }}
                    batches={[`cdcc-3`]}
                  />
                  <Heading
                    i18nKey='/heading/credits-and-deductions/care-providers/provider-expenses-people'
                    condition='/cdccMoreThanOneQualifyingPerson'
                    batches={[`cdcc-3`]}
                  />
                  <InfoDisplay
                    i18nKey='/info/credits-and-deductions/care-providers/provider-expenses'
                    condition={{ operator: `isFalse`, condition: `/hasReportedDependentCareBenefits` }}
                    batches={[`cdcc-0`]}
                  />
                  <InfoDisplay
                    i18nKey='/info/credits-and-deductions/care-providers/provider-expenses-reported-benefits'
                    condition='/hasReportedDependentCareBenefits'
                    batches={[`cdcc-0`]}
                  />
                  <InfoDisplay
                    i18nKey='/info/credits-and-deductions/care-providers/provider-expenses-exclude'
                    conditions={[
                      `/hasQpWhoTurned13AndWasNotUnableToCareForSelf`,
                      { operator: `isFalse`, condition: `/hasMoreThanOneQpWhoTurned13AndWasNotUnableToCareForSelf` },
                    ]}
                    batches={[`cdcc-3`]}
                  />
                  <InfoDisplay
                    i18nKey='/info/credits-and-deductions/care-providers/provider-expenses-exclude-people'
                    condition='/hasMoreThanOneQpWhoTurned13AndWasNotUnableToCareForSelf'
                    batches={[`cdcc-3`]}
                  />
                  <Dollar path='/cdccCareProviders/*/writableAmountPaidForCare' batches={[`cdcc-2`]} />
                  <SaveAndOrContinueButton />
                </Screen>
              </SubSubcategory>
            </Gate>
          </SubSubcategory>
        </CollectionLoop>
      </SubSubcategory>
    </Gate>
  );
};
