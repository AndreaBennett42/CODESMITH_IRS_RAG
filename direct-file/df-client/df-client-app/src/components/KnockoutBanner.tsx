import { Alert } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import { useKnockoutCheck } from '../hooks/useKnockoutCheck.js';
import { useFlow } from '../flow/flowConfig.js';
import { useMemo } from 'react';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import InternalLink from './InternalLink/index.js';
import { Condition } from '../flow/Condition.js';
import EndOfFilingSeasonBanner from './EndOfFilingSeasonBanner/EndOfFilingSeasonBanner.js';
import { useLocation } from 'react-router-dom';

const KnockoutBanner = () => {
  const { t } = useTranslation(`translation`);
  const { getIsKnockedOut } = useKnockoutCheck();
  const { factGraph } = useFactGraph();
  const flow = useFlow();
  const location = useLocation();
  const knockouts = useMemo(() => flow.screens.filter((screen) => screen.isKnockout), [flow]);
  const isTooLate = new Condition(`/isTooLateToFileOrResubmit`).evaluate(factGraph, null);

  const firstActiveKnockout = knockouts.find((screen) =>
    // Knockouts are weird, where their conditions are always context-free (no collection item ID),
    // and there's no good way today to get the collection item that triggered the knockout, so
    // we just evaluate the knockout condition without looking at the rest of the conditions
    // so we can work around the lack of collection item ID
    new Condition(screen.conditions.slice(-1)[0]).evaluate(factGraph, null)
  );

  if (!getIsKnockedOut() || !firstActiveKnockout) return <EndOfFilingSeasonBanner />;

  const internalLinkRoute = isTooLate
    ? `/flow/you-and-your-family/about-you/tax-deadline-ko`
    : firstActiveKnockout?.fullRoute(null);

  // If we're on the knockout explanation page, don't show the banner
  // NOTE: Avoiding includes here to prevent not showing the KO on screens with a substring of the KO
  // disregard query params for the comparison
  if (internalLinkRoute.split(`?`)[0] === location.pathname) return null;

  const knockoutMessage = isTooLate ? `banner.knockout.deadline` : `banner.knockout.mainMessage`;

  return (
    <section aria-label={t(`banner.knockout.name`)}>
      <Alert type='error' headingLevel='h6' role='alert' className='usa-alert--emergency'>
        <InternalLink route={internalLinkRoute} i18nKey={knockoutMessage} collectionId={null} />
      </Alert>
    </section>
  );
};

export default KnockoutBanner;
