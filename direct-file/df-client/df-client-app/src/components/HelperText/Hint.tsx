import { InfoDisplayProps } from '../../types/core.js';
import { buildHintKey } from '../FormControl/index.js';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';
import { I18nKey } from '../Translation/Translation.js';
import { Path } from '../../fact-dictionary/Path.js';
import { CommonTranslation } from 'df-i18n';

export type HintProps = Omit<InfoDisplayProps, 'gotoNextScreen' | `i18nKey`> & {
  hintId: string;
  i18nKey: I18nKey;
};

// Use for short helper text that has style 'usa-hint' (slightly grayed out)
const Hint = ({ hintId, i18nKey, collectionId }: HintProps) => {
  let hintKey = Array.isArray(i18nKey)
    ? i18nKey.map(CommonTranslation.getNamespacedKey)
    : [CommonTranslation.getNamespacedKey(i18nKey)];
  const { i18n } = useTranslation();
  /* currently hints are all paths or keys that have
   their content in the info namespace */
  if (!hintKey.some((key) => i18n.exists(key))) {
    //check if i18nKey passed can be built into a hint key
    hintKey = hintKey.map((key) => buildHintKey(key as Path));

    if (!hintKey.some((key) => i18n.exists(key))) return null;
  }
  return (
    <div id={hintId} className='usa-hint'>
      <Translation i18nKey={hintKey} collectionId={collectionId} />
    </div>
  );
};

export default Hint;
