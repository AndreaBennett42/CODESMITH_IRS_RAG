package gov.irs.directfile.api.pdf;

import java.util.Map;

import static java.util.Map.entry;

@SuppressWarnings("PMD.AvoidDuplicateLiterals")
public enum PdfLanguages {
    EN(
            "en",
            "en-US",
            "",
            "",
            Map.ofEntries(
                    entry("biologicalChild", "Child"),
                    entry("adoptedChild", "Child"),
                    entry("stepChild", "Stepchild"),
                    entry("fosterChild", "Foster Child"),
                    entry("grandChildOrOtherDescendantOfChild", "Grandchild"),
                    entry("childInLaw", "Child's Spouse"),
                    entry("sibling", "Sibling"),
                    entry("childOfSibling", "Sibling Child"),
                    entry("halfSibling", "Sibling"),
                    entry("childOfHalfSibling", "Sibling Child"),
                    entry("stepSibling", "Sibling"),
                    entry("childOfStepSibling", "Sibling Child"),
                    entry("otherDescendantOfSibling", "Sibling Child"),
                    entry("otherDescendantOfHalfSibling", "Sibling Child"),
                    entry("otherDescendantOfStepSibling", "Sibling Child"),
                    entry("siblingsSpouse", "Sibling Spouse"),
                    entry("siblingInLaw", "Spouse Sibling"),
                    entry("parent", "Parent"),
                    entry("fosterParent", "Parent"),
                    entry("siblingOfParent", "Parent Sibling"),
                    entry("grandParent", "Grandparent"),
                    entry("otherAncestorOfParent", "Grandparent"),
                    entry("stepParent", "Parent"),
                    entry("parentInLaw", "Spouse Parent"),
                    entry("noneOfTheAbove", "Other")),
            Map.of(true, "yes", false, "no")),
    ES(
            "es",
            "es-ES",
            " (Spanish Version)",
            " (sp)",
            Map.ofEntries(
                    entry("biologicalChild", "Hijo(a)"),
                    entry("adoptedChild", "Hijo(a)"),
                    entry("stepChild", "Hijastro(a)"),
                    entry("fosterChild", "Hijo(a) de crianza"),
                    entry("grandChildOrOtherDescendantOfChild", "Nieto(a)"),
                    entry("childInLaw", "Cónyuge del hijo(a)"),
                    entry("sibling", "Hermano(a)"),
                    entry("childOfSibling", "Hijo(a) del hermano(a)"),
                    entry("halfSibling", "Hermano(a)"),
                    entry("childOfHalfSibling", "Hijo(a) del hermano(a)"),
                    entry("stepSibling", "Hermano(a)"),
                    entry("childOfStepSibling", "Hijo(a) del hermano(a)"),
                    entry("otherDescendantOfSibling", "Hijo(a) del hermano(a)"),
                    entry("otherDescendantOfHalfSibling", "Hijo(a) del hermano(a)"),
                    entry("otherDescendantOfStepSibling", "Hijo(a) del hermano(a)"),
                    entry("siblingsSpouse", "Cónyuge del hermano(a)"),
                    entry("siblingInLaw", "Hermano(a) del cónyuge"),
                    entry("parent", "Padre o madre"),
                    entry("fosterParent", "Padre o madre"),
                    entry("siblingOfParent", "Hermano(a) de los padres"),
                    entry("grandParent", "Abuelo(a)"),
                    entry("otherAncestorOfParent", "Abuelo(a)"),
                    entry("stepParent", "Padre o madre"),
                    entry("parentInLaw", "Padres del cónyuge"),
                    entry("noneOfTheAbove", "Otro")),
            Map.of(true, "sí", false, "no"));

    private String code;
    private String pdfLanguage;
    private String pdfSubjectSuffix;
    private String pdfTitleSuffix;
    private Map<String, String> relationshipTranslations;
    private Map<Boolean, String> booleanTranslations;

    public static PdfLanguages fromString(String languageString) {
        // Disable false semgrep alert. `languageString` should already be sanitized by
        // now.
        //
        // nosemgrep: find_sec_bugs.IMPROPER_UNICODE-1
        if (languageString.toLowerCase().equals(ES.code)) {
            return ES;
        }
        return EN;
    }

    public String getCode() {
        return this.code;
    }

    public String getPdfLanguage() {
        return this.pdfLanguage;
    }

    public String getPdfSubjectSuffix() {
        return this.pdfSubjectSuffix;
    }

    public String getPdfTitleSuffix() {
        return this.pdfTitleSuffix;
    }

    public String translateRelationship(String relationshipOption) {
        String result = this.relationshipTranslations.get(relationshipOption);
        return result == null ? relationshipOption : result;
    }

    public String translateBoolean(boolean bool) {
        String result = this.booleanTranslations.get(bool);
        return result == null ? "" + bool : result;
    }

    PdfLanguages(
            String code,
            String pdfLanguage,
            String pdfSubjectSuffix,
            String pdfTitleSuffix,
            Map<String, String> relationshipTranslations,
            Map<Boolean, String> booleanTranslations) {
        this.code = code;
        this.pdfLanguage = pdfLanguage;
        this.pdfSubjectSuffix = pdfSubjectSuffix;
        this.pdfTitleSuffix = pdfTitleSuffix;
        this.relationshipTranslations = relationshipTranslations;
        this.booleanTranslations = booleanTranslations;
    }
}
