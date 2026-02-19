import { InfoDisplayProps } from '../types/core.js';
import ContentDisplay from './ContentDisplay/index.js';
import DownloadPDFButton from './DownloadPDFButton/DownloadPDFButton.js';

const InfoDisplay = ({
  i18nKey,
  inlinePDFButtonI18nKey,
  collectionId,
  borderBottom,
  context = {},
  ...contentProps
}: InfoDisplayProps) => {
  return (
    <ContentDisplay
      className={borderBottom ? `border-bottom` : ``}
      i18nKey={i18nKey}
      collectionId={collectionId}
      allowedTags={[`p`, `ul`, `ol`, `li`, `h2`, `h3`]}
      context={context}
      additionalComponents={
        inlinePDFButtonI18nKey && contentProps.taxId
          ? {
              InlinePDFButton: <DownloadPDFButton taxId={contentProps.taxId} i18nKey={inlinePDFButtonI18nKey} inline />,
            }
          : {}
      }
      {...contentProps}
    />
  );
};

export default InfoDisplay;
