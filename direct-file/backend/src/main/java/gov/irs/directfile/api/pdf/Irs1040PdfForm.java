package gov.irs.directfile.api.pdf;

import java.util.Optional;
import java.util.UUID;

import lombok.extern.slf4j.Slf4j;

import gov.irs.factgraph.types.BankAccount;

import gov.irs.directfile.api.pdf.load.PdfConfiguration;
import gov.irs.directfile.models.FactEvaluationResult;

@Slf4j
class Irs1040PdfForm extends PdfForm {

    public Irs1040PdfForm(final String templateName) {
        super(templateName);
        shouldRemoveTinDashes = true;
    }

    @Override
    protected boolean computePseudoFacts(
            final FactEvaluationResult facts, final UUID itemId, PdfConfiguration pdfConfig) {
        // When refund is due, "X" out missing account and routing numbers.
        // We use pseudo paths /^~pdf(Routing|Account)NumberOrXs$/ for this.
        String routingNumberFormValue = "";
        String accountNumberFormValue = "";
        if (facts.getBoolean("/dueRefund")) {
            final Optional<Object> opt = facts.getOptional("/xmlRefundBankAccount");
            if (opt.isEmpty()) {
                routingNumberFormValue = "X".repeat(9);
                accountNumberFormValue = "X".repeat(17);
            } else {
                final BankAccount account = (BankAccount) opt.get();
                routingNumberFormValue = account.routingNumber();
                accountNumberFormValue = account.accountNumber();
            }
        }
        facts.put("~pdfRoutingNumberOrXs", routingNumberFormValue);
        facts.put("~pdfAccountNumberOrXs", accountNumberFormValue);

        // Use pseudo path ~positiveSocialSecurityBenefitsOrEmpty to blank out negative values.
        String positiveSsBenefitsOrZero =
                facts.getOptional("/socialSecurityBenefits").orElse("").toString();
        if (positiveSsBenefitsOrZero.length() > 0) {
            try {
                if (Double.parseDouble(positiveSsBenefitsOrZero) < 0) {
                    positiveSsBenefitsOrZero = "";
                }
            } catch (NumberFormatException e) {
                log.warn("Unable to set social security benefits", e);
            }
        }
        facts.put("~positiveSocialSecurityBenefitsOrEmpty", positiveSsBenefitsOrZero);
        return true;
    }
}
