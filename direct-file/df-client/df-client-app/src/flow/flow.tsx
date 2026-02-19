/* eslint-disable max-len */
import { Category, Flow } from './flowDeclarations.js';

import { AboutYouSubcategory } from './flow-chunks/you-and-your-family/AboutYouSubcategory.js';
import { SpouseSubcategory } from './flow-chunks/you-and-your-family/SpouseSubcategory.js';
import { FamilyAndHHSubcategory } from './flow-chunks/you-and-your-family/FamilyAndHHSubcategory.js';
import { FilingStatusSubcategory } from './flow-chunks/you-and-your-family/FilingStatusSubcategory.js';
import { IncomeSourcesSubcategory } from './flow-chunks/income/IncomeSourcesSubcategory.js';
import {
  PreReleaseRetirementSubcategory,
  RetirementIncomeSubcategory,
} from './flow-chunks/income/RetirementIncomeSubcategory.js';
import { AlaskaPermanentFundSubcategory } from './flow-chunks/income/AlaskaPermanentFundSubcategory.js';
import { JobIncomeSubcategory } from './flow-chunks/income/JobIncomeSubcategory.js';
import { InterestIncomeSubcategory } from './flow-chunks/income/InterestIncomeSubcategory.js';
import { UnemploymentIncomeSubcategory } from './flow-chunks/income/UnemploymentIncomeSubcategory.js';
import { SocialSecurityIncomeSubcategory } from './flow-chunks/income/SocialSecurityIncomeSubcategory.js';
import { DeductionsSubcategory } from './flow-chunks/credits-and-deductions/DeductionsSubcategory.js';
import { DependentCareBenefitsSubcategory } from './flow-chunks/income/DependentCareBenefitsSubcategory.js';
import { HsaSubcategory } from './flow-chunks/income/HsaSubcategory.js';
import { CreditsSubcategory } from './flow-chunks/credits-and-deductions/CreditsSubcategory.js';
import { EstimatedTaxesSubcategory } from './flow-chunks/your-taxes/EstimatedTaxesSubcategory.js';
import { AmountSubcategory } from './flow-chunks/your-taxes/AmountSubcategory.js';
import { PaymentMethodSubcategory } from './flow-chunks/your-taxes/PaymentMethodSubcategory.js';
import { OtherPreferencesSubcategory } from './flow-chunks/your-taxes/OtherPreferencesSubcategory.js';
import { ReviewSubcategory } from './flow-chunks/sign-and-submit/ReviewSubcategory.js';
import { SignSubcategory } from './flow-chunks/sign-and-submit/SignSubcategory.js';
import { SubmitSubcategory } from './flow-chunks/sign-and-submit/SubmitSubcategory.js';
import { PrintAndMailSubcategory } from './flow-chunks/sign-and-submit/PrintAndMailSubcategory.js';
import { TotalIncomeSummarySubcategory } from './flow-chunks/income/TotalIncomeSummarySubcategory.js';

/**
 * Welcome to the flow!
 *
 * The flow, while currently existing as code, has an eventual goal of becoming configuration that
 * tax experts will be able to modify. This leads to a few design choices:
 *
 * 1. Everything in this file should remain serializable.
 * 2. While many of the below react components look like our fact components, they have separate
 *    definitions from the components that actually render.
 *      - The components here have type FC<ComponentNameDeclaration>.
 *      - The components that render have type FC<ComponentNameProps> - these props may have additional
 *        info and callbacks -- but their props always _extend_ the props that exist here.
 * 3. We have to maintain a mapping between the components we declare here in our config and the components
 *    that eventually render. Right now, the best place to understand some of that mapping is between this file,
 *    FlowDeclarations, ContentDeclarations, FlowConfig, getNextScreen and Screen.tsx.
 */
const flowNodes = (
  <Flow>
    <Category route='you-and-your-family'>
      {AboutYouSubcategory}
      {SpouseSubcategory}
      {FamilyAndHHSubcategory}
      {FilingStatusSubcategory}
    </Category>

    <Category route='income'>
      {IncomeSourcesSubcategory}
      {JobIncomeSubcategory}

      {UnemploymentIncomeSubcategory}
      {InterestIncomeSubcategory}
      {AlaskaPermanentFundSubcategory}
      {DependentCareBenefitsSubcategory}
      {HsaSubcategory}
      {PreReleaseRetirementSubcategory}
      {RetirementIncomeSubcategory}
      {SocialSecurityIncomeSubcategory}
      {/* This must always be the last subcategory in income */}
      {TotalIncomeSummarySubcategory}
    </Category>

    <Category route='credits-and-deductions'>
      {DeductionsSubcategory}
      {CreditsSubcategory}
    </Category>
    <Category route='your-taxes'>
      {EstimatedTaxesSubcategory}
      {AmountSubcategory}
      {PaymentMethodSubcategory}
      {OtherPreferencesSubcategory}
    </Category>

    <Category route='complete'>
      {ReviewSubcategory}
      {SignSubcategory}
      {PrintAndMailSubcategory}
      {SubmitSubcategory}
    </Category>
  </Flow>
);

export default flowNodes;
