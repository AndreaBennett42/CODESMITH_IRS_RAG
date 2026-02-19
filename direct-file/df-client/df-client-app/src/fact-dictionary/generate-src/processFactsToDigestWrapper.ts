/* eslint-disable @typescript-eslint/no-explicit-any */
// I wish this were more typed and readable, but the XML produces some heavily nested structures
// that are not worth writing interfaces for. The best way to understand this is to look

import { WrappedFact } from '@irs/js-factgraph-scala';

// at the generated input (facts.ts) and generated output (wrappedFacts.ts)
export default function processFactsToDigestWrapper(rawFacts: any[]): WrappedFact[] {
  return rawFacts.flatMap((rawFact: any) => {
    const writable = rawFact.Writable ? processWritable(rawFact.Writable) : null;
    const derived = rawFact.Derived ? processDerived(rawFact.Derived) : null;
    const placeholder = rawFact.Placeholder ? processDerived(rawFact.Placeholder) : null;
    return {
      path: rawFact[`@path`],
      writable,
      derived,
      placeholder,
    };
  });
}

function processOptions(rawNode: any): any {
  const attrs = Object.keys(rawNode).filter((key) => key.startsWith(`@`) || key === `#text`);
  const buffer: any = {};
  for (const attr of attrs) {
    const rawName = attr.startsWith(`@`) ? attr.replace(`@`, ``) : `value`;
    buffer[rawName] = rawNode[attr];
  }
  return buffer;
}

function processWritable(rawWriteable: any): any {
  const typeName = Object.keys(rawWriteable)[0];
  const defaultOptions = processOptions(rawWriteable[typeName]);
  const options = typeName === `CollectionItem` ? {} : defaultOptions;
  const collectionItemAlias = typeName === `CollectionItem` ? defaultOptions.collection : null;
  const limits = processLimits(rawWriteable[`Limit`]);
  return {
    typeName,
    options,
    collectionItemAlias,
    limits: limits,
  };
}

function processLimits(rawNode: any): any {
  if (rawNode !== undefined) {
    const rawNodes = Array.isArray(rawNode) ? rawNode : [rawNode];
    const limits = rawNodes.map((node: any) => {
      const typeName = Object.keys(node).filter((name) => !name.startsWith(`@`))[0];
      if (typeName === `Dependency`) {
        return processDependentLimit(node, typeName);
      }
      return {
        operation: node[`@type`],
        level: `Error`, // TODO: Figure out how levels are encoded
        node: {
          typeName,
          options: {
            value: String(node[typeName]),
          },
          children: [],
        },
      };
    });
    return limits;
  }
  return [];
}

function processDependentLimit(node: any, typeName: string): any {
  return {
    operation: node[`@type`],
    level: `Error`, // TODO: Figure out how levels are encoded
    node: {
      typeName,
      options: {
        path: String(node.Dependency[`@path`]),
      },
      children: [],
    },
  };
}

function processDerived(rawNode: any): any {
  const inner = (currentNode: any, typeName: string): any => {
    if (Array.isArray(currentNode)) {
      return currentNode.map((node: any) => inner(node, typeName));
    } else if (typeof currentNode === `object`) {
      const keys = Object.keys(currentNode);
      const defaultOptions = processOptions(currentNode);
      const hasValueChild = typeof currentNode[typeName] !== `object`;
      const options =
        hasValueChild && currentNode[typeName] ? { value: String(currentNode[typeName]) } : defaultOptions;
      const children = keys
        .filter((key) => !key.startsWith(`@`) && !key.startsWith(`#`))
        .flatMap((key) => inner(currentNode[key], key));
      return {
        typeName,
        options,
        children,
      };
    } else {
      const value = String(currentNode);
      return {
        typeName,
        options: { value },
        children: [],
      };
    }
  };
  const typeName = Object.keys(rawNode)[0];
  return inner(rawNode[typeName], typeName);
}
