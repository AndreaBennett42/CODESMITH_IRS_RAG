import { useFactGraph } from '../../../../factgraph/FactGraphContext.js';
import { getStateExportableFactsFromGraph } from '../../../../utils/exportUtils.js';
import { useMemo } from 'react';
import DisclosedExportableFacts from './DisclosedExportableFacts/DisclosedExportableFacts.js';

const useGetExportableFacts = () => {
  const { factGraph } = useFactGraph();
  return useMemo(() => getStateExportableFactsFromGraph(factGraph), [factGraph]);
};

const TransferableDataList = () => {
  const exportableFacts = useGetExportableFacts();

  return <DisclosedExportableFacts exportableFacts={exportableFacts} />;
};

export default TransferableDataList;
