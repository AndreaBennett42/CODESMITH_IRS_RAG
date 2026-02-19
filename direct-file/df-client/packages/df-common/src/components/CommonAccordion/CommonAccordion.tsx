import { FunctionComponent, useState } from 'react';
import { Accordion } from '@trussworks/react-uswds';
import { TranslationProps } from '../CommonContentDisplay/contentGenerator.js';
import { CommonContentDisplay } from '../CommonContentDisplay/CommonContentDisplay.js';

export interface CommonAccordionItemProps {
  title: React.ReactNode | string;
  content: React.ReactNode;
  expanded: boolean;
  id: string;
  className?: string;
  headingLevel: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
  handleToggle?: (event: React.MouseEvent<HTMLButtonElement>) => void;
}

export type CommonAccordionProps = {
  i18nKey: string;
  TranslationComponent: FunctionComponent<TranslationProps>;
  collectionId?: string | null;
  contentOverride?: React.ReactNode;
  allowedTags?: string[] | undefined;
  asExpanded?: boolean;
  headingLevel?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
  additionalComponents?: Record<string, JSX.Element>;
  className?: string;
};

function CommonAccordion({
  i18nKey,
  TranslationComponent,
  collectionId = null,
  contentOverride,
  allowedTags,
  asExpanded = false,
  className,
  additionalComponents,
  headingLevel,
}: CommonAccordionProps) {
  const [expanded, setExpanded] = useState(asExpanded);
  const accordionHeading = headingLevel ? headingLevel : `h3`;
  const items: CommonAccordionItemProps[] = [
    {
      id: i18nKey,
      title: <TranslationComponent i18nKey={`${i18nKey}.heading`} collectionId={collectionId} />,
      content: contentOverride ? (
        contentOverride
      ) : (
        <CommonContentDisplay
          i18nKey={i18nKey}
          collectionId={collectionId}
          additionalComponents={additionalComponents}
          TranslationComponent={TranslationComponent}
          allowedTags={allowedTags}
        />
      ),
      expanded: expanded,
      headingLevel: accordionHeading,
      handleToggle: () => setExpanded((previousValue) => !previousValue),
    },
  ];

  return <Accordion className={className} bordered items={items} />;
}

export default CommonAccordion;
