import { Grid } from '@trussworks/react-uswds';
import Translation from '../Translation/Translation.js';
import IntroContent from '../IntroContent/IntroContent.js';
import { isPostFederalFilingDeadline } from '../../utils/dateUtils.js';

export type CardInfo = {
  headingKey: string;
  additionalInfoKey: string;
  bodyKey: string;
};

export const REMINDER_CARD_BEFORE_TAX_DAY: CardInfo = {
  headingKey: `reminderTaxReturnCard.heading`,
  additionalInfoKey: `reminderTaxReturnCard.additionalInfo`,
  bodyKey: `reminderTaxReturnCard`,
};

export const REMINDER_CARD_AFTER_TAX_DAY: CardInfo = {
  headingKey: `reminderTaxReturnCardAfterFilingDeadline.heading`,
  additionalInfoKey: ``,
  bodyKey: `reminderTaxReturnCardAfterFilingDeadline`,
};

export const QUESTIONS: CardInfo = {
  headingKey: `reminderTaxReturnCardQuestions.heading`,
  additionalInfoKey: ``,
  bodyKey: `reminderTaxReturnCardQuestions`,
};

export const QuestionsReminderCard = () => {
  return <SimpleTaxReturnCard {...QUESTIONS} />;
};

export const SimpleTaxReturnCard = (cardInfo: CardInfo) => {
  return (
    <Grid col={12} className='border-base-lighter margin-top-3 border-2px shadow-2'>
      <div className='padding-205'>
        <h3 className='margin-0 padding-bottom-2'>
          {<Translation i18nKey={cardInfo.headingKey} collectionId={null} />}
        </h3>
        <div className='usa-prose'>
          <Translation i18nKey={cardInfo.additionalInfoKey} collectionId={null} />
          <IntroContent i18nKey={cardInfo.bodyKey} collectionId={null} />
        </div>
      </div>
    </Grid>
  );
};

const SimpleReminderTaxReturnCard = () => {
  const currentDate = new Date();

  let cardType: CardInfo;
  if (isPostFederalFilingDeadline(currentDate)) {
    cardType = REMINDER_CARD_AFTER_TAX_DAY;
  } else {
    cardType = REMINDER_CARD_BEFORE_TAX_DAY;
  }

  return <SimpleTaxReturnCard {...cardType} />;
};

export default SimpleReminderTaxReturnCard;
