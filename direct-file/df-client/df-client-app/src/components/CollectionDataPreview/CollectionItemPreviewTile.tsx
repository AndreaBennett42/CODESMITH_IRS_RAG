import { useTranslation } from 'react-i18next';
import Translation from '../Translation/index.js';
import { DOLLAR_FORMAT_PARAMS } from '../../misc/factGraphPrettyPrint.js';
import { z } from 'zod';
import { CollectionDataPreviewVariant } from './collectionDataPreviewHelpers.js';
import { i18n } from 'i18next';
import { w2Schema } from '../../redux/slices/data-import/schema/W2Schema.js';
import { interestIncomeSchema } from '../../redux/slices/data-import/schema/InterestIncomeSchema.js';

export type CollectionItemProps = z.infer<typeof interestIncomeSchema> &
  z.infer<typeof w2Schema> & {
    id: string;
  };

type CollectionItemPreviewTileProps = {
  id: string;
  collectionContext: string;
  collection: CollectionItemProps;
};

export const getCollectionItemContent = (i18n: i18n, collectionContext: string, collection: CollectionItemProps) => {
  switch (collectionContext) {
    case CollectionDataPreviewVariant.FORM_W2: {
      const DOLLAR_FORMATTER = new Intl.NumberFormat(i18n.language, DOLLAR_FORMAT_PARAMS);
      const formattedDollarAmount = DOLLAR_FORMATTER.format(parseInt(collection.wagesTipsOtherCompensation));
      return {
        header: collection.employersAddress.nameLine,
        value3: formattedDollarAmount,
      };
    }
    case CollectionDataPreviewVariant.INTEREST_REPORT: {
      const DOLLAR_FORMATTER = new Intl.NumberFormat(i18n.language, DOLLAR_FORMAT_PARAMS);
      const dollarValue = collection.box1 ? parseInt(collection.box1) : 0;
      const formattedDollarAmount = dollarValue ? DOLLAR_FORMATTER.format(dollarValue) : ``;
      return {
        header: collection.payerName,
        value3: formattedDollarAmount,
      };
    }
    default:
      throw new Error(`No values set for ${collectionContext}`);
  }
};

export const CollectionItemPreviewTile = ({ id, collectionContext, collection }: CollectionItemPreviewTileProps) => {
  const { t, i18n } = useTranslation();
  const baseItemI18nKey = `datapreviews.${collectionContext}.collectionListing`;
  const { value3 } = getCollectionItemContent(i18n, collectionContext, collection);

  return (
    <div className='padding-left-4' id={id}>
      <p>
        <Translation i18nKey={`${baseItemI18nKey}.label1`} collectionId={null} />
      </p>
      <p>
        <strong>{t(`${baseItemI18nKey}.label2`)}</strong>
        <span className='display-block'>
          <Translation i18nKey={`${baseItemI18nKey}.value2`} collectionId={collection.id} />
        </span>
      </p>
      <p>
        <strong>{t(`${baseItemI18nKey}.label3`)}</strong>
        <span className='display-block'>
          {i18n.exists(`${baseItemI18nKey}.value3`) ? (
            <Translation i18nKey={`${baseItemI18nKey}.value3`} collectionId={collection.id} />
          ) : (
            value3
          )}
        </span>
      </p>
    </div>
  );
};
