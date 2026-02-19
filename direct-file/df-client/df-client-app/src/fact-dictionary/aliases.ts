// HACKHACK it would be better if this were generated from the fact dictionary

// A slice is specific item out of a collection, which may itself be an alias
export const w2Slices = [`/formW2WithAllocatedTips/`] as const;

// Sometimes you just need a single path off a collection item slice or alias,
// and it's the only one that really even makes sense to use in the app. For those
// cases, you can just add the path you need here
export const aliasPaths = [
  `/formW2WithBox12Knockout/knockoutBox12Code`,
  `/formW2WithBox14YonkersKnockout/knockoutBox14YonkersCode`,
  `/formW2WithBox14NYRRTAKnockout/knockoutBox14NYRRTACode`,
  `/formW2WithBox12Knockout/knockoutBox12Code`,
  `/formW2WithBox20YonkersKnockoutCode/locality`,
] as const;

export const filerCollectionAliases = [
  `/filersWithTins/`,
  `/filersMaybeEligibleForDisability/`,
  `/filersQualifiedForEdcThroughDisability/`,
  `/filersWithHsa/`,
  `/filersWithHsaContributions/`,
  `/filersWithHsaCanHaveDistribution/`,
  `/filersRequiredToFileForm8889/`,
  `/cdccQualifyingFilers/`,
] as const;

// These will get transformed to look like `/filers/*/fullName`
export const individualFilerAliases = [
  `/secondaryFiler/`,
  `/primaryFiler/`,
  `/pdfSecondaryFiler/`,
  `/exemptSpouse/`,
  `/form1099Rs/*/filer/`,
  `/formW2s/*/filer/`,
  `/interestReports/*/filer/`,
  `/form1099Gs/*/filer/`,
  `/socialSecurityReports/*/filer/`,
  `/form1099Miscs/*/filer/`,
  `/form1099Rs/*/filer/`,
  `/hsaDistributions/*/filer/`,
  ...w2Slices.map((slice) => `${slice}filer/`),
] as const;

export const dependentCollectionAliases = [
  `/qualifyingDependentsCollection/`,
  `/claimedDependentsCollection/`,
  `/unclaimedDependentsCollection/`,
  `/eitcDependentsCollection/`,
  `/odcDependentsCollection/`,
  `/ctcDependentsCollection/`,
  `/form8862CtcDependentsCollection/`,
  `/form8862OdcDependentsCollection/`,
  `/form8862EitcDependentsCollection/`,
  `/unclaimedEITCQcsCollection/`,
  `/familyAndHouseholdWithTins/`,
  `/hohQualifyingPeople/`,
  `/cdccQualifyingPeople/`,
  `/qssQualifyingDependentCollection/`,
  `/qssQualifyingUnclaimedCollection/`,
  `/scheduleEicDependents/`,
  `/potentialScheduleEicDependents/`,
  `/cdccNonDependentQualifyingPeopleAssignedTins/`,
  `/cdccNonDependentQualifyingPeopleTinsPinsNeeded/`,
  `/cdccNonDependentQualifyingPeopleTinsPinsSaved/`,
  `/cdccNonFilerQpsWhoWereUnableToCareForSelfAndNotQcUnderAge13/`,
  `/cdccNonFilerQpsWhoTurned13InTaxYear/`,
  `/cdccNonFilerQpsWhoTurned13InTaxYearAbleToCareForSelf/`,
  `/cdccNonFilerQcsWhoTurned13InTaxYearUnableToCareForSelf/`,
] as const;

export const hsaDistributionsAliases = [`/primaryFilerHsaDistributions/`, `/secondaryFilerHsaDistributions/`] as const;

export const form1099GAliases = [
  `/primaryFiler1099Gs/`,
  `/primaryFiler1099GsWithAmount/`,
  `/secondaryFiler1099Gs/`,
  `/secondaryFiler1099GsWithAmount/`,
] as const;

export const interestReportsAliases = [
  `/primaryFilerInterestReports/`,
  `/primaryFilerInterestReportsWith1099Amount/`,
  `/primaryFilerInterestReportsWithNo1099Amount`,
  `/secondaryFilerInterestReports/`,
  `/secondaryFilerInterestReportsWith1099Amount/`,
  `/secondaryFilerInterestReportsWithNo1099Amount/`,
] as const;

export const form1099MiscAliases = [`/alaskaPfd1099s/`] as const;

export const form1099RAliases = [
  `/fullyTaxable1099Rs/`,
  `/notFullyTaxable1099Rs/`,
  `/form1099RsWithDisabilityCodeOnly/`,
  `/form1099RsWithTaxFreePensionRollover/`,
  `/form1099RsWithPartlyTaxablePension/`,
  `/form1099RsWithFullyTaxablePension/`,
  `/form1099RsWithPensionRollover/`,
  `/form1099RsWithDirectRollover/`,
  `/form1099RsWithUnsupportedDistributionCode/`,
  `/primaryFiler1099Rs/`,
  `/primaryFilerForm1099RsWithDisabilityCodeOnly/`,
  `/secondaryFiler1099Rs/`,
  `/secondaryFilerForm1099RsWithDisabilityCodeOnly/`,
  `/secondaryFiler1099RsReportedOnSaversCredit/`,
  `/primaryFiler1099RsReportedOnSaversCredit/`,
] as const;

export const w2Aliases = [`/secondaryFilerW2s/`, `/primaryFilerW2s/`] as const;

export const dependentCollectionAliasesWithoutTrailingSlash = dependentCollectionAliases.map((dc) =>
  dc.substring(0, dc.length - 1)
);

export const filerCollectionAliasesWithoutTrailingSlash = filerCollectionAliases.map((dc) =>
  dc.substring(0, dc.length - 1)
);

export const filerAliasesWithoutTrailingSlash = individualFilerAliases.map((a) => a.substring(0, a.length - 1));

export const individualDependentAliases = [
  `/firstQualifyingDependent/`,
  `/xmlQualifyingHOHPerson/`,
  `/hohQualifyingPerson/`,
  `/firstHohQP/`,
  `/xmlFirstQssQualifyingUnclaimedDependent/`,
  `/dependentWhoWasBornAndDiedInTaxYearWithoutTin/`,
  `/dependentWhoCausedNonCustodialKnockoutSplit/`,
  `/dependentWhoCausedContradictory8832Knockout/`,
] as const;

export const individualDependentAliasesWithoutTrailingSlash = individualDependentAliases.map((a) =>
  a.substring(0, a.length - 1)
);

export const individualW2Aliases = [`/oneAndOnlyW2/`, `/cdccCareProviders/*/employerWhoFurnishedCare/`] as const;
export const individualW2AliasesWithoutTrailingSlash = individualW2Aliases.map((a) => a.substring(0, a.length - 1));

export const bankAccountFields = [`/bankAccount`, `/xmlRefundBankAccount`, `/xmlPaymentBankAccount`] as const;
export const addressFields = [
  `/address`,
  `/formW2s/*/address`,
  `/formW2s/*/employerAddress`,
  `/form1099Rs/*/payer/address`,
  `/form1099Rs/*/address`,
  `/cdccCareProviders/*/address`,
] as const;
export const filerFields = [`/fullName`] as const;
// We export this to test that these fields still exist in the fact dictionary,
// and we're not creating outdated types
export const generationFieldAliases = [
  ...bankAccountFields,
  ...addressFields,
  ...filerAliasesWithoutTrailingSlash,
  ...dependentCollectionAliasesWithoutTrailingSlash,
  ...individualDependentAliasesWithoutTrailingSlash,
  ...individualW2AliasesWithoutTrailingSlash,
  ...filerCollectionAliasesWithoutTrailingSlash,
] as const;
