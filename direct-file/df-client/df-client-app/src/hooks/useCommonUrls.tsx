import { CommonLinkRenderer } from 'df-common-link-renderer';
import { TFunction } from 'i18next';
import { useMemo } from 'react';

export function useCommonUrls(t: TFunction) {
  return useMemo(
    () =>
      Object.fromEntries(
        Object.entries<string>(t(`commonUrls`, { returnObjects: true }) as ArrayLike<string>).map(
          ([key, url]: [string, string]) => [`${key}Link`, <CommonLinkRenderer key={url} url={url} />] as const
        )
      ),
    [t]
  );
}
