import useFact from '../../hooks/useFact.js';
import { Path } from '../../flow/Path.js';
import {
  CardInfo,
  REMINDER_CARD_AFTER_TAX_DAY,
  REMINDER_CARD_BEFORE_TAX_DAY,
  SimpleTaxReturnCard,
} from './SimpleReminderTaxReturnCard.js';

const FactAwareReminderTaxReturnCard = () => {
  const [isAfterTaxDay] = useFact<boolean>(Path.concretePath(`/isAfterTaxDay`, null));

  let cardType: CardInfo;
  if (isAfterTaxDay) {
    cardType = REMINDER_CARD_AFTER_TAX_DAY;
  } else {
    cardType = REMINDER_CARD_BEFORE_TAX_DAY;
  }

  return <SimpleTaxReturnCard {...cardType} />;
};

export default FactAwareReminderTaxReturnCard;
