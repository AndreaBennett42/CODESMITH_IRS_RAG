import { useMemo } from 'react';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Condition } from '../flow/Condition.js';
import { ItemConfig } from '../components/ConditionalList/ConditionalList.js';
import { useFilterContentContext } from '../context/FilterContentContext.js';
import { ConcretePath, scalaListToJsArray } from '@irs/js-factgraph-scala';

export function useConditionalListItems<Item extends ItemConfig>(
  items: Item[],
  collectionId: string | null
): (Item & { collectionId: string | null })[] {
  type ItemWithCollectionId = Item & { collectionId: string | null };
  const { factGraph } = useFactGraph();
  const { shouldFilterContentBasedOnTaxState } = useFilterContentContext();
  return useMemo(() => {
    if (!shouldFilterContentBasedOnTaxState) {
      return items.map((i) => {
        return {
          ...i,
          collectionId,
        };
      });
    }
    const expandedItems: ItemWithCollectionId[] = items.flatMap((item) => {
      if (item.collection) {
        const result = factGraph.get(item.collection as ConcretePath);
        const collectionItems = result.complete ? (scalaListToJsArray(result.get.getItemsAsStrings()) as string[]) : [];
        return collectionItems.map((ci) => {
          return { ...item, collectionId: ci };
        });
      } else return { ...item, collectionId: collectionId };
    });
    return expandedItems.filter((item) => {
      if (item.conditions && item.conditions.length > 0) {
        const result = item.conditions.every((c) => new Condition(c).evaluate(factGraph, item.collectionId));
        return result;
      }
      return true;
    });
  }, [items, factGraph, collectionId, shouldFilterContentBasedOnTaxState]);
}
