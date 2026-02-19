import { ReactNode, useContext } from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';

import { CommonContentDisplay } from '@irs/df-common';
import { Icon } from '@trussworks/react-uswds';

import { ChatIcon, DFAccordion, Heading, Translation as T } from '../components/index.js';
import { PilotPhaseContext } from '../layouts/Providers.js';
import { baseRouterPath } from '../constants.js';

const Section = ({ children, id = undefined }: { children: ReactNode | ReactNode[]; id?: string }) => (
  <section id={id}>
    <div className='grid-container'>{children}</div>
  </section>
);

const Content = ({ i18nKey, noBody = true }: { i18nKey: string; noBody?: boolean }) => (
  <CommonContentDisplay i18nKey={i18nKey} noBody={noBody} TranslationComponent={T} />
);

const BlueCheckedItems = ({ children }: { children: ReactNode[] }) => (
  <ul className='grid-row usa-icon-list'>
    {children.map((child, index) => (
      <li key={index} className='grid-col-6 usa-icon-list__item'>
        <div className='text-blue usa-icon-list__icon' aria-hidden='true'>
          <Icon.Check />
        </div>
        <div className='usa-icon-list__content'>{child}</div>
      </li>
    ))}
  </ul>
);

const Intro = () => {
  const { t } = useTranslation(`translation`);
  const phase = useContext(PilotPhaseContext);
  const NEXT = `/state`;

  return (
    <>
      <Helmet>
        <title>{t(`pages.Landing.pageTitle`)}</title>
      </Helmet>
      <Section id='welcome'>
        <Heading splash>
          {(phase?.showInvitationalHero || phase?.showAfterDeadlineInvitationalHero) && (
            <Content i18nKey='pages.Landing.heading.invitational' />
          )}
          {phase?.showThankYouHero && <Content i18nKey='pages.Landing.heading.thank-you' />}
        </Heading>
        <div className='grid-row'>
          <div className='grid-col-12 desktop:grid-col-6 tablet:grid-col-6'>
            <p className='lead-in'>
              {phase?.showInvitationalHero && <Content i18nKey='pages.Landing.section.welcome.invitational.lead-in' />}
              {phase?.showAfterDeadlineInvitationalHero && (
                <Content i18nKey='pages.Landing.section.welcome.invitational.lead-in-after-deadline' />
              )}
            </p>
            <p className='lead-in margin-bottom-1'>
              <Content i18nKey='pages.Landing.section.welcome.invitational.checklist.header' />
            </p>
            <BlueCheckedItems>
              <Content i18nKey='pages.Landing.section.welcome.invitational.checklist.first' />
              <Content i18nKey='pages.Landing.section.welcome.invitational.checklist.second' />
              <Content i18nKey='pages.Landing.section.welcome.invitational.checklist.third' />
              <Content i18nKey='pages.Landing.section.welcome.invitational.checklist.fourth' />
            </BlueCheckedItems>
            <p className='lead-in'>
              <Content i18nKey='pages.Landing.section.welcome.invitational.second-paragraph' />
            </p>
          </div>

          {phase?.showLandingPageImage && (
            <div className='grid-col-12 desktop:grid-col-6 tablet:grid-col-6 margin-top-5 hidden-on-mobile'>
              <img alt='' className='padding-left-05' src={`${baseRouterPath}/imgs/blueperson_laptop.png`} />
            </div>
          )}
        </div>

        <div className='grid-row grid-gap'>
          <div className='grid-col-12 desktop:grid-col-8 margin-bottom-2'>
            {phase?.enableScreener && (
              <>
                <Link className='usa-button' to={NEXT}>
                  {t(`pages.Landing.button.text`)}
                </Link>
              </>
            )}
          </div>
        </div>
      </Section>

      {phase?.showDataImportContent && (
        <Section id='data-import'>
          <h2>
            <Content i18nKey='pages.Landing.section.dataImport.heading' />
          </h2>
          <div className='grid-row content-container'>
            <p>
              <Content i18nKey='pages.Landing.section.dataImport.description' />
            </p>
          </div>
        </Section>
      )}

      {phase?.showBenefitsSections && (
        <>
          <Section id='more-benefits'>
            <h2>
              <Content i18nKey='pages.Landing.section.more-benefits.heading' />
            </h2>
            <div className='grid-row content-container'>
              <div className='grid-col'>
                <Content i18nKey='pages.Landing.section.more-benefits.benefits' />
              </div>
            </div>
            <div className='grid-row faq'>
              {/** First column of accordions */}
              {phase.showOpenSeasonFAQs && (
                <div className='grid-col'>
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.state-tax' />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.learn-more' />
                </div>
              )}
              {phase.showEndOfFilingFAQs && (
                <div className='grid-col'>
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.need-to-file' />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.amend-return' />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.learn-more' />
                </div>
              )}
              {phase.showPostFilingSeasonFAQs && (
                <div className='grid-col'>
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.state-tax' />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.learn-more' />
                </div>
              )}
              {/** Second column of accordions */}
              {phase.showOpenSeasonFAQs && (
                <div className='grid-col'>
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.tax-questions' />
                  {phase.showDataImportContent && (
                    <DFAccordion
                      splash
                      i18nKey='pages.Landing.section.faq.data-import'
                      allowedTags={[`p`, `h4`, `ul`, `li`]}
                    />
                  )}
                </div>
              )}
              {phase.showEndOfFilingFAQs && (
                <div className='grid-col'>
                  <DFAccordion
                    splash
                    i18nKey='pages.Landing.section.faq.state-tax-maybe-late'
                    allowedTags={[`p`, `h4`]}
                  />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.rejected-still-time' />
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.where-is-refund' />
                  {phase.showDataImportContent && (
                    <DFAccordion
                      splash
                      i18nKey='pages.Landing.section.faq.data-import'
                      allowedTags={[`p`, `h4`, `ul`, `li`]}
                    />
                  )}
                </div>
              )}
              {phase.showPostFilingSeasonFAQs && (
                <div className='grid-col'>
                  <DFAccordion splash i18nKey='pages.Landing.section.faq.when-open' />
                  {phase.showDataImportContent && (
                    <DFAccordion
                      splash
                      i18nKey='pages.Landing.section.faq.data-import'
                      allowedTags={[`p`, `h4`, `ul`, `li`]}
                    />
                  )}
                </div>
              )}
            </div>
          </Section>
          <Section id='benefits'>
            <h2>
              <Content i18nKey='pages.Landing.section.benefits.heading' />
            </h2>
            <div className='feature-list'>
              <div className='container'>
                <div className='margin-right-2'>
                  <ChatIcon />
                </div>
                <div className='flex-5'>
                  <p>
                    <Content i18nKey='pages.Landing.section.benefits.quote-1' />
                  </p>
                </div>
              </div>
              <div className='container tablet:margin-left-2 desktop:grid-offset-1'>
                <div className='margin-right-2'>
                  <ChatIcon />
                </div>
                <div className='flex-5'>
                  <p>
                    <Content i18nKey='pages.Landing.section.benefits.quote-2' />
                  </p>
                </div>
              </div>
            </div>
          </Section>
        </>
      )}
      {phase?.showAfterDeadlineSections && (
        <>
          <Section id='after-deadline'>
            <h2>
              <Content i18nKey='pages.Landing.section.after-deadline.heading' />
            </h2>
            <p className='margin-bottom-5'></p>
            <div className='grid-row after-deadline grid-gap'>
              {/** First column of accordions */}
              <div className='grid-col'>
                <DFAccordion i18nKey='pages.Landing.section.after-deadline.get-copies-of-return' />
                <DFAccordion i18nKey='pages.Landing.section.after-deadline.amend-return' />
              </div>
              <div className='grid-col'>
                <DFAccordion i18nKey='pages.Landing.section.after-deadline.rejected-return' />
                <DFAccordion i18nKey='pages.Landing.section.after-deadline.disaster-postponed-deadline' />
              </div>
            </div>
          </Section>
          <Section id='help-with-taxes'>
            <h2>
              <Content i18nKey='pages.Landing.section.help-with-taxes.heading' />
            </h2>
            <p className='margin-bottom-5'></p>
            <div className='grid-row help-with-taxes grid-gap'>
              <div className='grid-col'>
                <DFAccordion i18nKey='pages.Landing.section.help-with-taxes.where-is-refund' />
                <DFAccordion i18nKey='pages.Landing.section.help-with-taxes.get-help-with-taxes' />
              </div>
              <div className='grid-col'>
                <DFAccordion i18nKey='pages.Landing.section.help-with-taxes.other-ways-to-file' />
                <DFAccordion i18nKey='pages.Landing.section.help-with-taxes.submission-after-deadline' />
              </div>
            </div>
          </Section>
        </>
      )}
    </>
  );
};

export default Intro;
