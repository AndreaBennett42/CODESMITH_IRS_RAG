import Translation from '../components/Translation/index.js';
import { CommonNotFound } from '@irs/df-common';

const NotFound = () => {
  const i18nKey = `notFound`;

  return <CommonNotFound i18nKey={i18nKey} TranslationComponent={Translation} />;
};

export default NotFound;
