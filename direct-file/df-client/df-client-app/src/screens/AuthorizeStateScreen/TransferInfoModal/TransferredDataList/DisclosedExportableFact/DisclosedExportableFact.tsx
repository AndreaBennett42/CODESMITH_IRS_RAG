import { ExportableFact } from '../../../../../utils/exportUtils.js';

import styles from './DisclosedExportableFact.module.scss';
import { useTranslation } from 'react-i18next';

import Translation from '../../../../../components/Translation/Translation.js';
import { useState } from 'react';
import { getFactLabelI18nKey } from '../utils.js';

type DisclosedExportableFactProps = {
  factKey: string;
  fact: ExportableFact;
  collectionKey?: string;
};
const DisclosedExportableFact = ({ factKey, fact, collectionKey }: DisclosedExportableFactProps) => {
  const { value, sensitive } = fact;

  const { t, i18n } = useTranslation(`translation`);
  const [show, setShow] = useState(!sensitive);

  // Render nothing if the fact does not have a value
  // Legally, we do not need to disclose the absence of information being transferred
  if (value === null) {
    return null;
  }

  const i18nKey = getFactLabelI18nKey(factKey, collectionKey);
  const i18nKeyExists = i18n.exists(i18nKey);
  const label = i18nKeyExists ? <Translation i18nKey={i18nKey} collectionId={null} /> : factKey;

  return (
    <>
      <dt className={styles.descriptionTerm}>{label}</dt>
      <dd className={styles.descriptionDefinition}>
        {fact.sensitive ? (
          <>
            {show ? value : [...Array(value.length)].map((_e) => ` •`)}&nbsp; (
            <button type='button' className='usa-button--unstyled' onClick={() => setShow((show: boolean) => !show)}>
              {t(`button.${show ? `hide` : `show`}`)}
            </button>
            )
          </>
        ) : (
          value
        )}
      </dd>
    </>
  );
};

export default DisclosedExportableFact;
