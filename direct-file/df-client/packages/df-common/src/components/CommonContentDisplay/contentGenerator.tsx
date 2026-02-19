import { ReactNode, createElement, cloneElement, FC } from 'react';

export interface TranslationProps {
  i18nKey: string | string[];
  collectionId: string | null;
  context?: object;
  components?: Record<string, JSX.Element>;
}

function generateContent(
  i18nKey: string,
  body: string | object,
  collectionId: string | null,
  allowedTags: string[] | undefined,
  TranslationComponent: FC<TranslationProps>,
  components: Record<string, JSX.Element> = {},
  context: object
): ReactNode {
  const defaultTags = [`p`, `ul`, `ol`, `li`];
  allowedTags = allowedTags ? allowedTags : defaultTags;
  const componentsKeys = Object.keys(components);

  // This allows for a simple string body
  if (typeof body === `string`) {
    return (
      <TranslationComponent i18nKey={i18nKey} collectionId={collectionId} components={components} context={context} />
    );
  }

  // Handles "leaf" objects where content is a string
  const handleStringContent = (
    index: number,
    tag: string,
    fancyTag: JSX.Element | null,
    tagPath: string[],
    isArray: boolean
  ) => {
    const newTagPath = [...tagPath, tag];
    const i18nSubKey = i18nKey + newTagPath.join(`.`);
    const translatedElem = (
      <TranslationComponent
        i18nKey={i18nSubKey}
        key={index}
        collectionId={collectionId}
        components={components}
        context={context}
      />
    );

    // Strings can be provided "tagless" at the top level of an array
    // They will then be turned into paragraphs except for the first one in an array.
    if (isArray && tag === `0`) {
      return translatedElem; // return unwrapped first element
    } else if (isArray) {
      tag = `p`;
    }
    // return wrapped elements
    return fancyTag
      ? cloneElement(fancyTag, { key: index }, translatedElem)
      : createElement(tag, { key: index }, translatedElem);
  };

  // Recursively traverses the tree of yaml objects to generate the react nodes
  function traverseTree(tree: object | (object | string)[], tagPath: string[]): ReactNode {
    const isArray = Array.isArray(tree);

    // Loop through all entries in either array or object
    const result = Object.entries(tree).map((item, index) => {
      const [tag, content] = item;
      const fancyTag = componentsKeys.includes(tag) ? components[tag] : null;

      // Filter out disallowed tags
      if (!isArray && !allowedTags?.includes(tag)) {
        return null;
      }
      // Handle leaf nodes where content is a string to be passed to Translation
      if (typeof content === `string`) {
        return handleStringContent(index, tag, fancyTag, tagPath, isArray);
      }
      // If array, wrap in current tag
      else if (Array.isArray(content)) {
        const newTagPath = [...tagPath, tag];
        const subtreeOutput = traverseTree(content, newTagPath);
        return fancyTag
          ? cloneElement(fancyTag, { key: index }, subtreeOutput)
          : createElement(tag, { key: index }, subtreeOutput);
      }
      // If object, return the internal element (wrapped in tag if necessary)
      else if (typeof content === `object`) {
        const newTagPath = [...tagPath, tag];
        const subtreeOutput = traverseTree(content, newTagPath);
        if (isArray) {
          return subtreeOutput;
        }
        return fancyTag
          ? cloneElement(fancyTag, { key: index }, subtreeOutput)
          : createElement(tag, { key: index }, subtreeOutput);
      } else {
        return undefined;
      }
    });

    return result;
  }

  return traverseTree(body, [``]);
}

export { generateContent };
