import { TFunction, i18n as i18nType } from 'i18next';
import { ConcretePath, FactGraph } from '@irs/js-factgraph-scala';
import { useTranslation } from 'react-i18next';
import useTranslateWithFacts from '../../../hooks/useTranslateWithFacts.js';
import { Fragment } from 'react';
import { RawCondition, Condition } from '../../../flow/Condition.js';
import styles from './ChecklistSubcategory.module.scss';

export type DataItemConfig = {
  itemKey: string;
  conditions?: RawCondition[];
  condition?: RawCondition;
};

// mostly shadowing the i18next type
type SpecialObject = string | object | Array<string | object>;
type IndexableObject = { [key: string]: SpecialObject };

/* quick and dirty masking; leaving the last four digits visible */
const maskAccountNumbers = (text: string): string => {
  const accountNumber = text;
  const maskedAccountNumber = `*`.repeat(accountNumber.length - 4) + accountNumber.slice(-4);
  return maskedAccountNumber;
};

/* note that when the taxpayer is precisely breakeven, we don't show any subsection content */
const handleAmount = (item: IndexableObject, amountStatus: `owe` | `refund` | `even`): SpecialObject => {
  if (amountStatus === `owe`) {
    return (item as { owe: object }).owe;
  } else if (amountStatus === `refund`) {
    return (item as { refund: object }).refund;
  } else if (amountStatus === `even`) {
    return [];
  }
  return item;
};

const handlePaymentMethod = (
  item: IndexableObject,
  amountStatus: `owe` | `refund` | `even`,
  payViaAch: boolean,
  refundViaAch: boolean
): SpecialObject => {
  if (amountStatus === `owe` && payViaAch) {
    return (item as { owe: { withdraw: object } }).owe.withdraw;
  } else if (amountStatus === `owe` && !payViaAch) {
    return (item as { owe: { payAfter: object } }).owe.payAfter;
  } else if (amountStatus === `refund` && refundViaAch) {
    return (item as { refund: { ach: object } }).refund.ach;
  } else if (amountStatus === `refund` && !refundViaAch) {
    return (item as { refund: { check: object } }).refund.check;
  }
  return item;
};

const handleObjectAdjustments = (
  item: IndexableObject,
  path: string,
  amountStatus: `owe` | `refund` | `even`,
  payViaAch: boolean,
  refundViaAch: boolean
): SpecialObject => {
  if (path.includes(`checklist./flow/your-taxes/amount`)) {
    return handleAmount(item, amountStatus);
  } else if (path.includes(`checklist./flow/your-taxes/payment-method`)) {
    return handlePaymentMethod(item, amountStatus, payViaAch, refundViaAch);
  }
  return item;
};

const getPaymentDetails = (factGraph: FactGraph) => {
  const amountOwedResult = factGraph.get(`/finalTaxAmount` as ConcretePath);
  const amountOwed: number = amountOwedResult.complete ? amountOwedResult.get : 0;
  const amountRefundResult = factGraph.get(`/overpayment` as ConcretePath);
  const amountRefund: number = amountRefundResult.complete ? amountRefundResult.get : 0;
  const amountStatus: `owe` | `refund` | `even` =
    amountOwed > amountRefund ? `owe` : amountRefund > amountOwed ? `refund` : `even`;
  const payViaAchResult = factGraph.get(`/payViaAch` as ConcretePath);
  const payViaAch: boolean = payViaAchResult.complete ? payViaAchResult.get : false;
  const refundViaAchResult = factGraph.get(`/refundViaAch` as ConcretePath);
  const refundViaAch: boolean = refundViaAchResult.complete ? refundViaAchResult.get : false;
  return { amountStatus, payViaAch, refundViaAch };
};

// This is the old method of rendering data reveal, which is now mostly replaced by the new data items.
// But some subcategories still use this method.
const populateValues = (
  tBasic: TFunction,
  tFact: (path: string, collectionId: string | null) => string,
  i18n: i18nType,
  subcategoryRoute: string,
  factGraph: FactGraph
): { value: string; isPopulated: boolean }[] => {
  const values: { value: string; isPopulated: boolean }[] = [];

  if (!i18n.exists(`checklist.${subcategoryRoute}.body`)) {
    return values;
  }

  const { amountStatus, payViaAch, refundViaAch } = getPaymentDetails(factGraph);

  const processItem = (item: SpecialObject, path: string, factGraph: FactGraph) => {
    if (typeof item === `string`) {
      const variableMatches = item.match(/{{.*?}}/g) || [];
      let isPopulated = true;

      for (const variable of variableMatches) {
        const translatedVariable = tFact(variable, null);
        if (translatedVariable === ``) {
          isPopulated = false;
          break;
        }
      }
      if (isPopulated) {
        let interpolatedText = item;
        variableMatches.forEach((variable) => {
          let translatedVariable = tFact(variable, null);
          if (variable === `{{/bankAccount/accountType}}`) {
            translatedVariable = translatedVariable.toLowerCase();
          } else if (variable === `{{/bankAccount/accountNumber}}`) {
            translatedVariable = maskAccountNumbers(translatedVariable);
          }
          interpolatedText = interpolatedText.replace(variable, translatedVariable);
        });
        if (interpolatedText === `0.00` || interpolatedText === `$0.00`) {
          interpolatedText = tBasic(`dataviews.noneReported`);
        }

        values.push({ value: interpolatedText, isPopulated });
        return;
      }
    } else if (Array.isArray(item)) {
      return item;
    } else if (typeof item === `object` && item !== null) {
      item = handleObjectAdjustments(item as IndexableObject, path, amountStatus, payViaAch, refundViaAch);

      (Object.keys(item) as Array<keyof typeof item>).forEach((key) => {
        processItem((item as IndexableObject)[key], `${path}.${key}`, factGraph);
      });
    }
  };

  const maybeArray = tBasic(`checklist.${subcategoryRoute}.body`, { returnObjects: true });
  processItem(maybeArray, `checklist.${subcategoryRoute}.body`, factGraph);

  return values;
};

type DataRevealProps = {
  i18nKey: string;
  dataItems?: DataItemConfig[];
  factGraph: FactGraph;
  subcategoryRoute: string;
  collectionId: null | string;
};

// DataReveal is used to display the data items under each subcategory in the checklist.
// They are also used to display the data items under the inner loop inside a data view.
const DataReveal = ({ i18nKey, dataItems, factGraph, subcategoryRoute, collectionId }: DataRevealProps) => {
  const { t: tBasic, i18n } = useTranslation();
  const { t: tFact } = useTranslateWithFacts(factGraph);

  const showDataItems = (items: DataItemConfig[]) => {
    if (!items) return [];

    const visibleItems = items.filter((item) => {
      if (item.condition) {
        return new Condition(item.condition).evaluate(factGraph, collectionId);
      }
      if (item.conditions && item.conditions.length > 0) {
        const result = item.conditions.every((c) => new Condition(c).evaluate(factGraph, null));
        return result;
      }
      return true;
    });
    return visibleItems.map((item) => {
      return tFact(`${i18nKey}.${item.itemKey}`);
    });
  };

  // The old method was all custom using populateValues. This should probably be refactored to use the new data items.
  // But to preserve the custom behavior for some subcategories, this code switches to using populateValues.
  // Note: even one custom item will force all to be populated by populateValues. You cannot have a mix (yet).
  let values: { value: string; isPopulated: boolean }[] = [];
  if (!dataItems) {
    values = populateValues(tBasic, tFact, i18n, subcategoryRoute, factGraph);
  }

  return (
    <div data-testid='checklist-data-reveal'>
      {/* If dataItems are defined, we use the new data items method. */}
      {dataItems && (
        <>
          {showDataItems(dataItems).map((result, index) => (
            <div key={index} className={styles.checklistSubsectionContent}>
              {result}
            </div>
          ))}
        </>
      )}

      {!dataItems &&
        values.map(
          (item, index) =>
            item.isPopulated && (
              <div key={index} className={styles.checklistSubsectionContent}>
                {item.value.includes(`<br />`)
                  ? item.value.split(`<br />`).map((line, i) => {
                      /* Because in populateValues we're dealing with facts instead of yaml keys,
                  the values we get are strings instead of html (output from tFact instead of <Translation>),
                  so we have to do a transformation here when there are html tags in the string.
                  We make what's in front of and behind any <br /> tag separate lines and
                  render a <br /> tag between them, but not after the last line */
                      return (
                        <Fragment key={i}>
                          {line}
                          {i < item.value.split(`<br />`).length - 1 && <br />}
                        </Fragment>
                      );
                    })
                  : item.value}
              </div>
            )
        )}
    </div>
  );
};

export default DataReveal;
