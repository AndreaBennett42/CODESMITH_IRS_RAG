import { LanguageDefinition, LanguageSelector } from '@trussworks/react-uswds';
import { HTMLAttributes } from 'react';

export type CommonLanguageSelectorProps = {
  langs: LanguageDefinition[];
  displayLang?: string;
};

export const CommonLanguageSelector = ({
  langs,
  ...props
}: CommonLanguageSelectorProps & HTMLAttributes<HTMLDivElement>) => {
  return <LanguageSelector {...props} langs={langs} />;
};
