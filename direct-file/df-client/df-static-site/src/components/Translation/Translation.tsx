import { Trans, useTranslation } from 'react-i18next';

import { buildLinkComponents } from '@irs/df-common';
import { CommonTranslation } from 'df-i18n';
import { ModalRef } from '@trussworks/react-uswds';
import { FC } from 'react';

export interface TranslationProps {
  i18nKey: string | string[];
  context?: object;
  components?: Record<string, JSX.Element>;
}

const Translation: FC<TranslationProps & { modalRef?: React.RefObject<ModalRef> }> = ({ i18nKey, components }) => {
  const { t } = useTranslation();

  const { urls } = CommonTranslation.maybeUrls(t, i18nKey);

  const linkComponents = buildLinkComponents(urls);

  const allowedComponents = {
    italic: <i />,
    strong: <strong />,
    ul: <ul />,
    li: <li />,
    ol: <ol />,
    br: <br />,
    ...linkComponents,
    ...components,
  };
  return <Trans i18nKey={i18nKey} components={allowedComponents} />;
};

export default Translation;
