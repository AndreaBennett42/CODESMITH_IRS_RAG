import { MutableRefObject, createRef, MouseEvent } from 'react';

export const scrollToInvalidElement = (
  event: MouseEvent<HTMLElement>,
  refs: MutableRefObject<Map<string, MutableRefObject<HTMLElement>>>,
  prefersReducedMotion: boolean | undefined
) => {
  event.preventDefault(); // prevent a tag from reloading the page

  const { target } = event;
  const invalidElementPath = getInvalidElementPath(target);

  if (refs.current && invalidElementPath) {
    const thisRef = refs.current.get(invalidElementPath);
    if (thisRef && thisRef.current) {
      // Avoid animating the scroll when the user prefers reduced motion
      thisRef.current.focus({ preventScroll: !prefersReducedMotion });

      if (!prefersReducedMotion) {
        thisRef.current.scrollIntoView({
          behavior: `smooth`,
          block: `center`,
          inline: `center`,
        });
      }
    }
  }
};

const getInvalidElementPath = (target: EventTarget): string => {
  const targetElement = target as HTMLElement;

  if (targetElement.tagName.toLowerCase() === `a`) {
    return targetElement.getAttribute(`data-path`) || ``;
  } else if (
    targetElement.tagName.toLowerCase() === `strong` &&
    targetElement.parentElement?.tagName.toLowerCase() === `a`
  ) {
    return targetElement.parentElement.getAttribute(`data-path`) || ``;
  } else {
    throw new Error(
      // eslint-disable-next-line max-len
      `Either the data-path is not set for this fact, or some OTHER tag other than <strong> is being used in the field name. Please add this OTHER tag to this function: getInvalidElementPath in summaryHelper.ts`
    );
  }
};

export const handleRefFromRoute = (
  route: string,
  refMap: MutableRefObject<Map<string, MutableRefObject<HTMLElement>>>
) => {
  if (refMap && refMap.current.has(route)) {
    return refMap.current.get(route);
  } else {
    const ref = createRef<HTMLElement>();
    refMap?.current.set(route, ref as MutableRefObject<HTMLElement>);
    return ref;
  }
};
