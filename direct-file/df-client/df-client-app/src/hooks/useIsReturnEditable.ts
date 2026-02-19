import { useContext, useMemo } from 'react';
import { getTaxReturnById } from '../utils/taxReturnUtils.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Path } from '../flow/Path.js';

export const useIsReturnEditable = () => {
  const { factGraph } = useFactGraph();
  const { taxReturns, currentTaxReturnId } = useContext(TaxReturnsContext);
  const isEditable = useMemo(() => {
    const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
    const isTooLateToFileOrResubmit = factGraph.get(Path.concretePath(`/isTooLateToFileOrResubmit`, null)).get;
    const isReturnEditableOnBackend = !currentTaxReturn?.isEditable === false;
    return isReturnEditableOnBackend && !isTooLateToFileOrResubmit;
  }, [currentTaxReturnId, factGraph, taxReturns]);

  return { isReturnEditable: isEditable };
};
