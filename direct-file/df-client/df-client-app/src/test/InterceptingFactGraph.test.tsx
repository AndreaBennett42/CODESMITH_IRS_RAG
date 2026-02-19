import { filerWithZeroBalanceData } from './testData.js';
import { Path } from '../flow/Path.js';
import { DollarFactory } from '@irs/js-factgraph-scala';
import { setupFactGraph } from './setupFactGraph.js';

export const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;

describe(`InterceptingFactGraph`, () => {
  it(`Triggers the amount change flag when the final tax amount changes`, () => {
    const factGraph = setupFactGraph(filerWithZeroBalanceData).factGraph;

    const FINAL_TAX_AMOUNT_SEEN_FACT = Path.concretePath(`/flowHasSeenAmount`, null);
    const FINAL_TAX_AMOUNT_FACT = Path.concretePath(`/finalTaxAmount`, null);
    const AMOUNT_CHANGED_FACT = Path.concretePath(`/flowHasAmountChanged`, null);

    factGraph.set(FINAL_TAX_AMOUNT_SEEN_FACT, true);
    factGraph.save();

    expect(factGraph.get(FINAL_TAX_AMOUNT_SEEN_FACT).complete).toBeTruthy();
    expect(factGraph.get(FINAL_TAX_AMOUNT_SEEN_FACT).get).toBeTruthy();

    expect(factGraph.get(AMOUNT_CHANGED_FACT).complete).toBeFalsy();
    expect(factGraph.get(AMOUNT_CHANGED_FACT).get).toBeFalsy();

    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).complete).toBeTruthy();
    const startFinalTaxAmount = factGraph.get(FINAL_TAX_AMOUNT_FACT).get.toString();

    factGraph.set(Path.concretePath(`/formW2s/*/writableWages`, uuid), DollarFactory(`20000`).right);
    factGraph.save();

    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).complete).toBeTruthy();
    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).get.toString()).not.toBe(startFinalTaxAmount);

    expect(factGraph.get(AMOUNT_CHANGED_FACT).complete).toBeTruthy();
    expect(factGraph.get(AMOUNT_CHANGED_FACT).get).toBeTruthy();
  });
  it(`Doesn't trigger the amount change flag when the amount hasn't changed`, () => {
    const factGraph = setupFactGraph(filerWithZeroBalanceData).factGraph;

    const FINAL_TAX_AMOUNT_SEEN_FACT = Path.concretePath(`/flowHasSeenAmount`, null);
    const FINAL_TAX_AMOUNT_FACT = Path.concretePath(`/finalTaxAmount`, null);
    const AMOUNT_CHANGED_FACT = Path.concretePath(`/flowHasAmountChanged`, null);

    factGraph.set(FINAL_TAX_AMOUNT_SEEN_FACT, true);
    factGraph.save();

    expect(factGraph.get(FINAL_TAX_AMOUNT_SEEN_FACT).complete).toBeTruthy();
    expect(factGraph.get(FINAL_TAX_AMOUNT_SEEN_FACT).get).toBeTruthy();

    expect(factGraph.get(AMOUNT_CHANGED_FACT).complete).toBeFalsy();
    expect(factGraph.get(AMOUNT_CHANGED_FACT).get).toBeFalsy();

    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).complete).toBeTruthy();
    const startFinalTaxAmount = factGraph.get(FINAL_TAX_AMOUNT_FACT).get.toString();

    factGraph.set(Path.concretePath(`/formW2s/*/writableWages`, uuid), DollarFactory(`200000`).right);
    factGraph.save();

    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).complete).toBeTruthy();
    expect(factGraph.get(FINAL_TAX_AMOUNT_FACT).get.toString()).toBe(startFinalTaxAmount);

    expect(factGraph.get(AMOUNT_CHANGED_FACT).get).toBeFalsy();
  });
});
