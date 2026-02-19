import { useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { CommonContentDisplay } from '@irs/df-common';
import { Alert } from '@trussworks/react-uswds';

import {
  DFAccordion,
  Heading,
  Translation,
  Prose,
  NextLink,
  Breadcrumbs,
  StepIndicator,
  SubHeader,
} from '../../components/index.js';
import { PilotPhaseContext } from '../../layouts/Providers.js';

const State = () => {
  const { t } = useTranslation(`translation`);
  const navigate = useNavigate();
  const phase = useContext(PilotPhaseContext);
  const PREV = `/`;
  const NEXT = `/income`;
  if (!phase?.enableScreener) navigate(`/`);

  return (
    <>
      <Helmet>
        <title>{t(`pages.ScreenerState.pageTitle`)}</title>
      </Helmet>
      <SubHeader />
      <Breadcrumbs href={PREV} />
      <StepIndicator currentStepKey='ScreenerState' />
      <Prose>
        <Heading>{t(`pages.ScreenerState.heading`)}</Heading>
        <CommonContentDisplay i18nKey='pages.ScreenerState' TranslationComponent={Translation} />
        <DFAccordion
          i18nKey='pages.ScreenerState.Alaska'
          className='styled-state'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Alaska.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Arizona'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.Arizona.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Arizona.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.California'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.California.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Connecticut'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Connecticut.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Florida'
          allowedTags={[`p`, `ul`, `ol`, `li`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Florida.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Idaho'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.Idaho.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Idaho.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Illinois'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.Illinois.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Illinois.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Kansas'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Kansas.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Maine'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Maine.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Maryland'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.Maryland.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Maryland.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Massachusetts'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.Massachusetts.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Massachusetts.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Nevada'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Nevada.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.New Hampshire'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.New Hampshire.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.New Jersey'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.New Jersey.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.New Jersey.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.New Mexico'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.New Mexico.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.New Mexico.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.New York'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.New York.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.New York.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.North Carolina'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.North Carolina.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.North Carolina.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Oregon'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay i18nKey='pages.ScreenerState.Oregon.alert' TranslationComponent={Translation} />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Oregon.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Pennsylvania'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.Pennsylvania.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Pennsylvania.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.South Dakota'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{ link: <NextLink href={NEXT}>{t(`pages.ScreenerState.South Dakota.link`)}</NextLink> }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Tennessee'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{ link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Tennessee.link`)}</NextLink> }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Texas'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{ link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Texas.link`)}</NextLink> }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Washington'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{ link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Washington.link`)}</NextLink> }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Wisconsin'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `alert`, `link`]}
          className='styled-state'
          additionalComponents={{
            alert: (
              <Alert type={`info`} headingLevel={`h3`}>
                <CommonContentDisplay
                  i18nKey='pages.ScreenerState.Washington.alert'
                  TranslationComponent={Translation}
                />
              </Alert>
            ),
            link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Wisconsin.link`)}</NextLink>,
          }}
        />
        <DFAccordion
          i18nKey='pages.ScreenerState.Wyoming'
          allowedTags={[`p`, `ul`, `ol`, `li`, `h4`, `link`]}
          className='styled-state'
          additionalComponents={{ link: <NextLink href={NEXT}>{t(`pages.ScreenerState.Wyoming.link`)}</NextLink> }}
        />
        <Alert className='margin-top-3' type='warning' headingLevel='h3' validation>
          <CommonContentDisplay i18nKey='pages.ScreenerState.alert' TranslationComponent={Translation} />
        </Alert>
      </Prose>
      <NextLink href={NEXT}>{t(`pages.ScreenerState.button.text`)}</NextLink>
    </>
  );
};

export default State;
