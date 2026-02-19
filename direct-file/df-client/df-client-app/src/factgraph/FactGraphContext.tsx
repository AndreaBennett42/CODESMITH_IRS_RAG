import { ReactNode, createContext, useContext, useEffect } from 'react';
import { Path } from '../flow/Path.js';
import { save } from '../hooks/useApiHook.js';
import { InterceptingFactGraph } from './InterceptingFactGraph.js';
import { FactGraph } from '@irs/js-factgraph-scala';
import { TaxReturnSubmission, TaxReturnSubmissionStatus } from '../types/core.js';
import { FEDERAL_RETURN_STATUS } from '../constants/taxConstants.js';
import { useFlow } from '../flow/flowConfig.js';
import { CompleteFactGraphResult } from '@irs/js-factgraph-scala/src/typings/FactGraph.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import { is1099RFeatureFlagEnabled } from '../constants/pageConstants.js';
import { useAppDispatch } from '../redux/hooks.js';
import { resetElectronicSignatureFailure } from '../redux/slices/electronic-signature/electronicSignatureSlice.js';

// TODO: we could stop surrounding this in an unncessary object
// and just have FactGraphContextValue === FactGraph
interface FactGraphContextValue {
  factGraph: FactGraph;
}

export const FactGraphContext = createContext<FactGraphContextValue>({
  // I don't love this, but I love it more than always checking for undefined.
  // If you useFactGraph in a place without the context, you're gonna have a
  // runtime error.
  factGraph: undefined as unknown as FactGraph,
});

let instance: FactGraphContextValue | undefined = undefined;

type FactGraphContextProviderProps = {
  existingFacts?: object;
  children: ReactNode[] | ReactNode;
  forceNewInstance?: true; // triggering this props will cause 100s of ms of CPU time. Be careful with it.
  taxReturnSubmissions?: TaxReturnSubmission[];
  submissionStatus?: TaxReturnSubmissionStatus;
};

export const FactGraphContextProvider = ({
  existingFacts,
  forceNewInstance,
  taxReturnSubmissions = [],
  submissionStatus,
  children,
}: FactGraphContextProviderProps) => {
  const dispatch = useAppDispatch();
  const flow = useFlow();
  const { currentTaxReturnId } = useContext(TaxReturnsContext);
  const isResubmitting = taxReturnSubmissions.length > 0 && submissionStatus?.status === FEDERAL_RETURN_STATUS.REJECTED;
  if (!instance || forceNewInstance) {
    // we have to use a singleton instead of react lifecycle (e.g. useMemo) since we may navigate between different
    // pages (e.g. checklist and basescreen) that have different context providers.
    // We don't want to accidentally lose state if we've modified the fact graph but haven't yet saved to the backend.
    const factGraph = new InterceptingFactGraph(existingFacts, flow);

    instance = { factGraph };
  }
  // at this point it's always setup.
  const factGraph = instance.factGraph as FactGraph;

  useEffect(() => {
    const isResubmittingPath = Path.concretePath(`/isResubmitting`, null);
    const isResubmittingFactResult = factGraph.get(isResubmittingPath);
    const isResubmittingFactValue =
      isResubmittingFactResult.complete && (isResubmittingFactResult as CompleteFactGraphResult<boolean>).get;
    let needToSaveGraph = false;
    if (isResubmitting && !isResubmittingFactValue) {
      // We need to render flow content based on whether the taxpayer is resubmitting their return after a rejection.
      // If they are, we assign a value to the "/isResubmitting" fact, so that it can be used within the flow.
      factGraph.set(isResubmittingPath, true);
      needToSaveGraph = true;
      // Reset the electronic signing failure flag so that users can use ESSAR even if they had to use legacy
      // signing on their previous submission.
      dispatch(resetElectronicSignatureFailure());
    }
    if (needToSaveGraph) {
      // eslint-disable-next-line df-rules/no-factgraph-save
      factGraph.save();
    }
  }, [dispatch, factGraph, isResubmitting]);

  useEffect(() => {
    globalThis.loadFactGraph = (factGraphJson: string | object) => {
      const factObject = typeof factGraphJson === `string` ? JSON.parse(factGraphJson) : factGraphJson;
      const url = `${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${currentTaxReturnId}`;
      save(url, { body: { facts: factObject } }).then(() => {
        window.location.reload();
      });
    };

    if (process.env.NODE_ENV == `development`) {
      // If we're in dev mode and on the checklist, we can load a fact graph from a json string.
      // This operates by sending the fact graph to the backend, and then reloading the page to remove any local state.
      globalThis.saveFactGraphToLocalStorageKey = (keyId: string, force?: boolean) => {
        // eslint-disable-next-line eqeqeq
        if (localStorage.getItem(keyId) != null && !force) {
          return;
        }
        localStorage.setItem(keyId, factGraph.toJSON());
      };

      globalThis.loadFactGraphFromLocalStorageKey = (keyId: string) => {
        const jsonString = localStorage.getItem(keyId);
        if (jsonString) {
          globalThis.loadFactGraph(jsonString);
        } else {
          return;
        }
      };
      if (import.meta.env.VITE_ENABLE_FLAMINGO_TELESCOPE === `true`) {
        // This is large and only for dev debug so we only load it when explicitly requested
        import(`../fact-dictionary/generated/facts.js`).then(({ facts }) => {
          globalThis.rawFacts = facts;
        });
      }
    }
  }, [currentTaxReturnId, factGraph]);
  return <FactGraphContext.Provider value={instance}>{children}</FactGraphContext.Provider>;
};

export const useFactGraph = () => {
  return useContext(FactGraphContext);
};

