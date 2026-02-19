import { Subcategory, Screen } from '../../flowDeclarations.js';
import {
  ExitButton,
  Heading,
  InfoDisplay,
  SetFactAction,
  StateInfoCard,
  DFAlert,
  TaxReturnAlert,
  ConditionalList,
  DFAccordion,
} from '../../ContentDeclarations.js';

export const PrintAndMailSubcategory = (
  <Subcategory
    route='print-and-mail'
    completeIf='/flowHasSeenSignPaperFile'
    displayOnlyIf='/isPaperPath'
    skipDataView={true}
  >
    <Screen route='sign-paper-file' condition='/cannotFindPinOrAgi'>
      <TaxReturnAlert i18nKey={`/info/paper-filing`} type='warning' condition='/cannotFindPinOrAgi' headingLevel='h2'>
        <DFAccordion i18nKey='/info/paper-filing-explainer' />
      </TaxReturnAlert>
      <Heading i18nKey='/heading/complete/sign-and-submit/sign-paper-file' />
      <InfoDisplay i18nKey='/info/complete/sign-and-submit/sign-paper-file-intro' />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/sign-paper-file-steps'
        inlinePDFButtonI18nKey='button.inlineDownloadPDF'
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-check'
        conditions={[`/dueRefund`, { operator: `isFalse`, condition: `/refundViaAch` }]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-direct-deposit'
        conditions={[`/dueRefund`, `/refundViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-now'
        conditions={[`/owesBalance`, `/payViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later'
        conditions={[
          `/owesBalance`,
          { operator: `isFalse`, condition: `/payViaAch` },
          { operator: `isFalse`, condition: `/isAfterTaxDay` },
        ]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later-after-tax-day'
        conditions={[`/owesBalance`, { operator: `isFalse`, condition: `/payViaAch` }, `/isAfterTaxDay`]}
        borderBottom
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-with-credit'
        stateLinki18nKey='/info/complete/sign-and-submit/state-with-credit-link'
        conditions={[`/hasStateFilingIntegration`, `/scopedStateDoesNotHavePersonalIncomeTax`]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-can-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[`/hasStateFilingIntegration`, `/stateCanTransferData`]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-cannot-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[
          `/hasStateFilingIntegration`,
          { operator: `isFalse`, condition: `/stateCanTransferData` },
          { operator: `isFalse`, condition: `/scopedStateDoesNotHavePersonalIncomeTax` },
        ]}
      />
      <SetFactAction path='/flowHasSeenSignPaperFile' source='/flowTrue' />
      <ExitButton />
    </Screen>
    <Screen route='sign-spouse-paper-file' condition='/spouseCannotFindPinOrAgi'>
      <TaxReturnAlert
        i18nKey={`/info/spouse-paper-filing`}
        type='warning'
        condition='/spouseCannotFindPinOrAgi'
        headingLevel='h2'
      />
      <Heading i18nKey='/heading/complete/sign-and-submit/sign-paper-file' />
      <InfoDisplay i18nKey='/info/complete/sign-and-submit/sign-paper-file-intro' />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/sign-paper-file-steps'
        inlinePDFButtonI18nKey='button.inlineDownloadPDF'
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-check'
        conditions={[`/dueRefund`, { operator: `isFalse`, condition: `/refundViaAch` }]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-direct-deposit'
        conditions={[`/dueRefund`, `/refundViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-now'
        conditions={[`/owesBalance`, `/payViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later'
        conditions={[
          `/owesBalance`,
          { operator: `isFalse`, condition: `/payViaAch` },
          { operator: `isFalse`, condition: `/isAfterTaxDay` },
        ]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later-after-tax-day'
        conditions={[`/owesBalance`, { operator: `isFalse`, condition: `/payViaAch` }, `/isAfterTaxDay`]}
        borderBottom
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-can-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[`/hasStateFilingIntegration`, `/stateCanTransferData`]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-cannot-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[
          `/hasStateFilingIntegration`,
          { operator: `isFalse`, condition: `/stateCanTransferData` },
          { operator: `isFalse`, condition: `/scopedStateDoesNotHavePersonalIncomeTax` },
        ]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-with-credit'
        stateLinki18nKey='/info/complete/sign-and-submit/state-with-credit-link'
        conditions={[`/hasStateFilingIntegration`, `/scopedStateDoesNotHavePersonalIncomeTax`]}
      />
      <SetFactAction path='/flowHasSeenSignPaperFile' source='/flowTrue' />
      <ExitButton />
    </Screen>
    <Screen route='sign-paper-file-ip-pin' condition='/isPaperPathDueToMissingIpPin'>
      <DFAlert
        headingLevel='h2'
        i18nKey={`/info/paper-filing/ip-pin-not-ready`}
        type='warning'
        condition='/isPaperPathDueToMissingIpPin'
      >
        <ConditionalList
          i18nKey='/info/paper-filing/ip-pin-not-ready'
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
        <DFAccordion i18nKey='/info/paper-filing/ip-pin-not-ready-explainer' />
      </DFAlert>
      <Heading i18nKey='/heading/complete/sign-and-submit/sign-paper-file' />
      <InfoDisplay i18nKey='/info/complete/sign-and-submit/sign-paper-file-intro' />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/sign-paper-file-steps'
        inlinePDFButtonI18nKey='button.inlineDownloadPDF'
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-check'
        conditions={[`/dueRefund`, { operator: `isFalse`, condition: `/refundViaAch` }]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/refund-info-direct-deposit'
        conditions={[`/dueRefund`, `/refundViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-now'
        conditions={[`/owesBalance`, `/payViaAch`]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later'
        conditions={[
          `/owesBalance`,
          { operator: `isFalse`, condition: `/payViaAch` },
          { operator: `isFalse`, condition: `/isAfterTaxDay` },
        ]}
        borderBottom
      />
      <InfoDisplay
        i18nKey='/info/complete/sign-and-submit/tax-owed-pay-later-after-tax-day'
        conditions={[`/owesBalance`, { operator: `isFalse`, condition: `/payViaAch` }, `/isAfterTaxDay`]}
        borderBottom
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-with-credit'
        stateLinki18nKey='/info/complete/sign-and-submit/state-with-credit-link'
        conditions={[`/hasStateFilingIntegration`, `/scopedStateDoesNotHavePersonalIncomeTax`]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-can-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[`/hasStateFilingIntegration`, `/stateCanTransferData`]}
      />
      <StateInfoCard
        i18nKey='/info/complete/sign-and-submit/state-taxes-cannot-transfer-data'
        stateLinki18nKey='/info/complete/sign-and-submit/state-taxes-link'
        conditions={[
          `/hasStateFilingIntegration`,
          { operator: `isFalse`, condition: `/stateCanTransferData` },
          { operator: `isFalse`, condition: `/scopedStateDoesNotHavePersonalIncomeTax` },
        ]}
      />
      <SetFactAction path='/flowHasSeenSignPaperFile' source='/flowTrue' />
      <ExitButton />
    </Screen>
  </Subcategory>
);
