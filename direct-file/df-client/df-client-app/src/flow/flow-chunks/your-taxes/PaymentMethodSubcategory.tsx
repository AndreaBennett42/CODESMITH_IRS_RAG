/* eslint-disable max-len */
import { Assertion, Gate, Screen, Subcategory, SubSubcategory } from '../../flowDeclarations.js';
import {
  Address,
  BankAccount,
  Boolean,
  ContextHeading,
  DatePicker,
  DFModal,
  DFAlert,
  Heading,
  IconList,
  InfoDisplay,
  SaveAndOrContinueButton,
  TaxReturnAlert,
  SetFactAction,
  ConditionalList,
  DFAccordion,
} from '../../ContentDeclarations.js';

export const PaymentMethodSubcategory = (
  <Subcategory
    route='payment-method'
    completeIf='/paymentSectionComplete'
    displayOnlyIf={{ operator: `isFalse`, condition: `/zeroBalance` }}
  >
    <Assertion
      type='info'
      i18nKey='dataviews./flow/your-taxes/payment-method.accountNumbersAssertion'
      conditions={[`/refundViaAch`, `/dueRefund`]}
    />
    <Assertion
      type='info'
      i18nKey='dataviews./flow/your-taxes/payment-method.accountNumbersAssertion'
      conditions={[`/owesBalance`, `/payViaAch`]}
    />
    <Gate condition='/dueRefund'>
      <Screen route='refund-intro'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/your-taxes/payment-method' />
        <Heading i18nKey='/heading/your-taxes/payment-method/refund-intro' />
        <InfoDisplay i18nKey='/info/your-taxes/payment-method/refund-intro' />
        <DFModal i18nKey='/info/your-taxes/payment-method/refund-apply-to-estimated-taxes' />
        <SaveAndOrContinueButton />
      </Screen>
      <SubSubcategory route='refund-method'>
        <Screen route='refund-method-choice'>
          <Heading i18nKey='/heading/your-taxes/payment-method/refund-method-choice' />
          <Boolean path='/refundViaAch' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='refund-mailing-address'>
        <Assertion
          type='info'
          i18nKey='dataviews./flow/your-taxes/payment-method.mailingAddressAssertion'
          conditions={[`/dueRefund`, { operator: `isFalse`, condition: `/refundViaAch` }]}
        />
        <Screen route='refund-mailing-address' condition={{ operator: `isFalse`, condition: `/refundViaAch` }}>
          <Heading i18nKey='/heading/your-taxes/payment-method/refund-mailing-address' />
          <InfoDisplay i18nKey='/info/your-taxes/payment-method/refund-mailing-address' />
          <Address path='/address' displayOnlyOn='data-view' hintKey='/info/why-cant-i-change-country' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <SubSubcategory route='bank-account-info'>
        <Screen route='refund-direct-deposit' condition='/refundViaAch'>
          <DFModal i18nKey='/info/your-taxes/payment-method/refund-split' />
          <Heading i18nKey='/heading/your-taxes/payment-method/refund-direct-deposit' />
          <BankAccount path='/bankAccount' />
          <DFModal i18nKey='/info/your-taxes/payment-method/routing-and-account-numbers' />
          <SaveAndOrContinueButton />
        </Screen>
      </SubSubcategory>
      <Screen route='refund-outro'>
        <Heading i18nKey='/heading/your-taxes/payment-method/refund-outro' />
        <InfoDisplay i18nKey='/info/your-taxes/payment-method/refund-outro' />
        <DFAlert
          i18nKey='/info/your-taxes/payment-method/refund-outro-verify-bank'
          headingLevel='h3'
          type='info'
          condition='/refundViaAch'
        />
        <DFAlert
          i18nKey='/info/your-taxes/payment-method/refund-outro-verify-address'
          headingLevel='h3'
          type='info'
          condition={{ operator: `isFalse`, condition: `/refundViaAch` }}
        />
        <IconList i18nKey='/iconList/your-taxes/payment-method/refund-outro/direct-deposit' condition='/refundViaAch' />
        <IconList
          i18nKey='/iconList/your-taxes/payment-method/refund-outro/mailing-address'
          condition={{ operator: `isFalse`, condition: `/refundViaAch` }}
        />
        <SaveAndOrContinueButton />
      </Screen>
    </Gate>
    <Gate condition='/owesBalance'>
      <Screen route='payment-intro'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/your-taxes/payment-method' />
        <Heading i18nKey='/heading/your-taxes/payment-method/payment-intro' />
        <InfoDisplay
          i18nKey='/info/your-taxes/payment-method/payment-intro'
          condition={{ operator: `isFalseOrIncomplete`, condition: `/isPaperPath` }}
        />
        <SaveAndOrContinueButton />
      </Screen>
      <SubSubcategory route='payment-method'>
        <Screen route='payment-method-choice-autocorrect' condition='/isPaperPathAndWantedToPayViaAch'>
          <TaxReturnAlert
            condition='/isPaperPathAndWantedToPayViaAch'
            type='error'
            i18nKey='/info/your-taxes/payment-method/payment-method-choice-autocorrect'
          />
          <Heading i18nKey='/heading/your-taxes/payment-method/payment-method-choice-autocorrect' />
          <InfoDisplay i18nKey='/info/your-taxes/payment-method/payment-method-choice-autocorrect' />
          <SetFactAction path='/flowHasSeenPaymentMethodAutocorrect' source='/flowTrue' />
          <SetFactAction path='/payViaAch' source='/flowFalse' />
          <SaveAndOrContinueButton />
        </Screen>
        <Gate condition='/isPaperPath'>
          <Screen
            route='payment-paper-path-assertion'
            condition={{ operator: `isFalseOrIncomplete`, condition: `/isPaperPathAndWantedToPayViaAch` }}
            actAsDataView={true}
          >
            <DFAlert
              headingLevel='h2'
              i18nKey={`/info/paper-filing/ip-pin-not-ready`}
              type='warning'
              conditions={[
                `/isPaperPathDueToMissingIpPin`,
                { operator: `isTrueAndComplete`, condition: `/isPaperPathAndNoPaymentChoiceOrWasAutocorrected` },
              ]}
            >
              <ConditionalList
                i18nKey={`/info/paper-filing/ip-pin-not-ready`}
                items={[
                  {
                    itemKey: `primary`,
                    conditions: [`/primaryFilerIsMissingIpPin`],
                    editRoute: `/flow/you-and-your-family/about-you/about-you-ip-pin-ready`,
                  },
                  {
                    itemKey: `secondary`,
                    conditions: [`/secondaryFilerIsMissingIpPin`],
                    editRoute: `/flow/you-and-your-family/spouse/spouse-mfj-ip-pin-ready`,
                  },
                  {
                    itemKey: `dependents`,
                    conditions: [`/oneOrMoreDependentsAreMissingIpPin`],
                    editRoute: `/flow/you-and-your-family/dependents/qualified-dependent-ip-pin-ready`,
                    collection: `/familyAndHousehold`,
                  },
                ]}
              />
              <DFAccordion
                i18nKey='/info/paper-filing/ip-pin-not-ready-explainer'
                internalLink='/flow/you-and-your-family/about-you/about-you-ip-pin-ready'
              />
            </DFAlert>
            <DFAlert
              headingLevel='h2'
              i18nKey={`/info/paper-filing`}
              type='warning'
              conditions={[
                `/cannotFindPinOrAgi`,
                { operator: `isTrueAndComplete`, condition: `/isPaperPathAndNoPaymentChoiceOrWasAutocorrected` },
              ]}
            >
              <DFAccordion i18nKey='/info/paper-filing-explainer' />
            </DFAlert>
            <DFAlert
              headingLevel='h2'
              i18nKey={`/info/spouse-paper-filing`}
              type='warning'
              conditions={[
                `/spouseCannotFindPinOrAgi`,
                { operator: `isTrueAndComplete`, condition: `/isPaperPathAndNoPaymentChoiceOrWasAutocorrected` },
              ]}
            >
              <DFAccordion i18nKey='/info/spouse-paper-filing-explainer' />
            </DFAlert>
            <Heading i18nKey='/heading/your-taxes/payment-method/payment-paper-path-assertion' />
            <InfoDisplay i18nKey='/info/paper-path-payment/paper-path-info' />
            <SetFactAction
              condition={{ operator: `isTrueOrIncomplete`, condition: `/payViaAch` }}
              path='/payViaAch'
              source='/flowFalse'
            />
            <SetFactAction path='/flowHasSeenPaymentPaperPathAssertion' source='/flowTrue' />
            <SaveAndOrContinueButton />
          </Screen>
        </Gate>
      </SubSubcategory>
      <Gate
        condition={{ operator: `isFalseOrIncomplete`, condition: `/isPaperPathAndHasSeenPaymentPaperPathAssertion` }}
      >
        <SubSubcategory route='payment-method'>
          <Screen route='payment-method-choice'>
            <Heading i18nKey='/heading/your-taxes/payment-method/payment-method-choice' />
            <InfoDisplay i18nKey='/info/your-taxes/payment-method/payment-method-due-date' />
            <Boolean path='/payViaAch' />
            <SetFactAction path='/payViaAch' condition='/isPaperPath' source='/flowFalse' />
            <DFModal i18nKey='/info/your-taxes/payment-method/payment-method-choice' />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <SubSubcategory route='bank-account-info'>
          <Screen route='payment-direct-debit' condition='/payViaAch'>
            {/* Pre-submission */}
            <TaxReturnAlert
              conditions={[
                `/achPaymentDateInThePast`,
                { operator: `isFalseOrIncomplete`, condition: `/isResubmitting` },
              ]}
              i18nKey='/info/your-taxes/payment-method/ach-payment-date-invalid'
              type='error'
            />
            <TaxReturnAlert
              conditions={[
                `/achPaymentDateAfterDeadline`,
                { operator: `isFalseOrIncomplete`, condition: `/isResubmitting` },
              ]}
              i18nKey='/info/your-taxes/payment-method/ach-payment-date-invalid'
              type='error'
            />

            {/* Post-submission */}
            <TaxReturnAlert
              conditions={[`/achPaymentDateInThePast`, `/isResubmitting`]}
              i18nKey='/info/your-taxes/payment-method/ach-payment-date-invalid-post-rejection'
              type='error'
            />
            <TaxReturnAlert
              conditions={[
                `/achPaymentDateAfterDeadline`,
                `/isResubmitting`,
                { operator: `isFalse`, condition: `/achPaymentDateMustBeToday` },
              ]}
              i18nKey='/info/your-taxes/payment-method/ach-payment-date-invalid-post-rejection'
              type='error'
            />
            <TaxReturnAlert
              conditions={[`/achPaymentDateMustBeToday`, `/isResubmitting`, `/achPaymentDateAfterDeadline`]}
              i18nKey='/info/your-taxes/payment-method/ach-payment-date-require-today'
              type='error'
            />

            <Heading i18nKey='/heading/your-taxes/payment-method/payment-direct-debit' />
            <BankAccount path='/bankAccount' />
            <DFModal i18nKey='/info/your-taxes/payment-method/routing-and-account-numbers' />
            <DatePicker
              condition={{ operator: `isFalseOrIncomplete`, condition: `/achPaymentDateMustBeToday` }}
              path='/achPaymentDate'
              labelledBy='legend'
              lastAllowableDatePath='/achPaymentDateLastAllowableDate'
              disallowPastDates={true}
            />
            <DatePicker
              condition='/achPaymentDateMustBeToday'
              path='/achPaymentDate'
              hintKey='/info/achPaymentDateToday'
              labelledBy='legend'
              lastAllowableDatePath='/achPaymentDateLastAllowableDate'
              disallowPastDates={true}
            />
            <SaveAndOrContinueButton />
          </Screen>
        </SubSubcategory>
        <Screen route='payment-another-way' condition={{ operator: `isFalse`, condition: `/payViaAch` }}>
          <Heading i18nKey='/heading/your-taxes/payment-method/payment-another-way' />
          <InfoDisplay i18nKey='/info/your-taxes/payment-method/payment-another-way' />
          <SaveAndOrContinueButton />
        </Screen>
        <Screen route='payment-outro'>
          <Heading i18nKey='/heading/your-taxes/payment-method/payment-outro' />
          <InfoDisplay i18nKey='/info/your-taxes/payment-method/payment-outro' />
          <DFAlert
            i18nKey='/info/your-taxes/payment-method/payment-outro'
            headingLevel='h3'
            type='warning'
            condition='/payViaAch'
          />
          <IconList i18nKey='/iconList/your-taxes/payment-method/payment-outro' condition='/payViaAch' />
          <IconList
            i18nKey='/iconList/your-taxes/payment-method/payment-another-way-outro'
            condition={{ operator: `isFalse`, condition: `/payViaAch` }}
          />
          <InfoDisplay
            i18nKey='/info/your-taxes/payment-method/payment-another-way-outro'
            condition={{ operator: `isFalse`, condition: `/payViaAch` }}
          />
          <SaveAndOrContinueButton />
        </Screen>
      </Gate>
    </Gate>
  </Subcategory>
);
