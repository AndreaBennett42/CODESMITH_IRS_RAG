import { CollectionFactory, ConcretePath, EmailAddress, TinFactory } from '@irs/js-factgraph-scala';
import { InterceptingFactGraph } from '../factgraph/InterceptingFactGraph.js';
import { Path } from '../flow/Path.js';
import { v4 as uuidv4 } from 'uuid';
import { wrappedFacts } from '../fact-dictionary/generated/wrappedFacts.js';
import { AbsolutePath } from '../fact-dictionary/Path.js';
import { createCollectionWrapper } from './persistenceWrappers.js';
import { paths } from '../fact-dictionary/generated/paths.js';
import { hasOwn } from '../utils/polyfills.js';

const writableFactPaths = wrappedFacts.filter(({ writable }) => !!writable).map(({ path }) => path as AbsolutePath);

function existingFactsHasEmail(existingFacts: object) {
  return hasOwn(existingFacts, `/email`);
}

export function setupFactGraph(existingFacts = {}) {
  const factGraph = new InterceptingFactGraph(existingFacts);

  for (const path of Object.keys(existingFacts)) {
    const absolutePath = Path.fromConcretePath(path as ConcretePath);

    if (!paths.includes(absolutePath)) {
      throw new Error(`Existing facts include unrecognized fact '${path}`);
    }

    // Flag any attempts to write to unwritable paths
    if (!writableFactPaths.includes(absolutePath)) {
      throw new Error(`Existing facts included non-writable fact ${path}`);
    }

    // Flag any invalid collection ids in paths
    if (Path.isAbstract(absolutePath)) {
      const collectionIdMatches = path.match(/#[0-9a-z-]+/g);
      if (!collectionIdMatches) {
        throw new Error(`Collection path appears malformed.  Unable to identify collection ID in path '${path}'`);
      } else if (collectionIdMatches.length > 1) {
        throw new Error(`Collection path appears malformed.  Found multiple collection IDs '${path}'`);
      } else {
        // Get the actual UUID, minus the # sign
        const collectionId = collectionIdMatches[0].slice(1);
        const [prefix] = absolutePath.split(`/*/`);

        const allCollectionIds = (existingFacts as Record<string, ReturnType<typeof createCollectionWrapper>>)[prefix]
          .item.items;

        if (!allCollectionIds.includes(collectionId)) {
          throw new Error(`Collection id in path '${path}' was not found in collection '${prefix}'`);
        }
      }
    }
  }

  const existingFactIssues = factGraph
    .checkPersister()
    .map((issue) => `${issue.message} \`${issue.path}\``)
    .join(`\n`);
  if (existingFactIssues.length > 0) {
    throw new Error(`The test data does not align with the fact dictionary ...\n${existingFactIssues}`);
  }

  const hasPrimaryFiler = factGraph.get(Path.concretePath(`/primaryFiler`, null)).complete;
  if (!hasPrimaryFiler) {
    const collections = wrappedFacts.filter((f) => f.writable?.typeName === `Collection`).map((f) => f.path);
    collections.map((c) => factGraph.set(c, CollectionFactory([])));

    const filersPath = Path.concretePath(`/filers`, null);
    const primaryFilerId = uuidv4();
    const secondaryFilerId = uuidv4();
    factGraph.set(filersPath, CollectionFactory([primaryFilerId, secondaryFilerId]));

    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save(); // required save before setting `isPrimaryFiler` in the collection
    factGraph.set(Path.concretePath(`/filers/*/isPrimaryFiler`, primaryFilerId), true);
    factGraph.set(Path.concretePath(`/filers/*/isPrimaryFiler`, secondaryFilerId), false);
    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save();
  }

  const emailFact = factGraph.get(Path.concretePath(`/email`, null));
  if (!emailFact.complete) {
    factGraph.set(Path.concretePath(`/email`, null), EmailAddress(`user.0000@example.com`));
    factGraph.save();
  }

  const tinFact = factGraph.get(Path.concretePath(`/primaryFiler`, null));
  if (!tinFact.complete) {
    console;
    factGraph.set(Path.concretePath(`/primaryFiler/tin`, null), TinFactory(`123456789`));
    factGraph.save();
  }

  return { factGraph };
}

/**
 * Deprecated! Legacy tests expect that we don't have tin/primary filers in our fact graph
 * but that isn't an accurate representation of the world. Try to use the above setup whenever possible.
 */
export function setupFactGraphDeprecated(existingFacts = {}) {
  const factGraph = new InterceptingFactGraph(existingFacts);
  const numExistingFacts = Object.keys(existingFacts).length;

  // These conditions are totally legacy but there are 100s of tests that rely on them existing like this.
  //
  // This basically ensures that there are two filers and one is primary which is also what the backend now does
  // in every environment.
  if (numExistingFacts === 0 || (numExistingFacts === 1 && existingFactsHasEmail(existingFacts))) {
    const collections = wrappedFacts.filter((f) => f.writable?.typeName === `Collection`).map((f) => f.path);
    collections.map((c) => factGraph.set(c, CollectionFactory([])));

    const filersPath = Path.concretePath(`/filers`, null);
    const primaryFilerId = uuidv4();
    const secondaryFilerId = uuidv4();
    factGraph.set(filersPath, CollectionFactory([primaryFilerId, secondaryFilerId]));

    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save(); // required save before setting `isPrimaryFiler` in the collection
    factGraph.set(Path.concretePath(`/filers/*/isPrimaryFiler`, primaryFilerId), true);
    factGraph.set(Path.concretePath(`/filers/*/isPrimaryFiler`, secondaryFilerId), false);
    // eslint-disable-next-line df-rules/no-factgraph-save
    factGraph.save();
  }

  const emailFact = factGraph.get(Path.concretePath(`/email`, null));
  if (!emailFact.complete) {
    factGraph.set(Path.concretePath(`/email`, null), EmailAddress(`user.0000@example.com`));
    factGraph.save();
  }

  return { factGraph };
}
