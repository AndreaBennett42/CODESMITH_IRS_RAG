import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { useFlow } from '../flow/flowConfig.js';
import { useNavigate } from 'react-router-dom';
import getNextScreen from '../screens/getNextScreen.js';
import { useCallback } from 'react';

export function useKnockoutCheck() {
  const { factGraph } = useFactGraph();
  const navigate = useNavigate();
  const flow = useFlow();
  const getIsKnockedOut = useCallback(() => {
    const isKnockedOutFact = factGraph.get(`/flowIsKnockedOut` as ConcretePath);
    return isKnockedOutFact.complete && isKnockedOutFact.get;
  }, [factGraph]);

  const redirectUserIfKnockedOut = useCallback(() => {
    // We need to call getIsKnockedOut() since redirectUserIfKnockedOut can be called
    // in a callback that modified the fact graph, in which case isKnockedOut might be stale
    if (getIsKnockedOut()) {
      const firstScreen = flow.screens[0]; // We need literally any screen
      const knockoutRoute = getNextScreen(firstScreen, factGraph, null, flow).routable.fullRoute(null);
      navigate(knockoutRoute);
    }
  }, [factGraph, flow, getIsKnockedOut, navigate]);
  return { getIsKnockedOut, redirectUserIfKnockedOut };
}
