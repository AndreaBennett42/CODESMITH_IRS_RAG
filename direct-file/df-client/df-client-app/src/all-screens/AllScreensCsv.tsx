import { Button } from '@trussworks/react-uswds';
import { useMemo } from 'react';
import getUrlSitemap, { UrlEntry } from '../utils/urlSitemap.js';
import { useFlow } from '../flow/flowConfig.js';
import styles from './AllScreens.module.scss';
import { useTranslation } from 'react-i18next';

export const AllScreensCsv = () => {
  const flow = useFlow();
  const { t } = useTranslation();

  // Get sitemap for CSV view
  const urls: UrlEntry[] = useMemo(() => {
    return getUrlSitemap(flow, t);
  }, [t, flow]);
  let csv: string[] = [];
  csv.push(`site, type, url, categoryTitle, subcategoryTitle, pageTitle, uniqueTitle`);
  const urlRows = urls.map(({ site, type, url, categoryTitle, subcategoryTitle, pageTitle, uniqueTitle }) => {
    return `${site},${type},${url},"${categoryTitle}","${subcategoryTitle}","${pageTitle}","${uniqueTitle}"`;
  });
  csv = csv.concat(urlRows);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(csv.join(`\n`));
    } catch (error) {
      return;
    }
  };

  return (
    <>
      <div>
        <Button type={`button`} onClick={handleCopy}>
          Copy CSV
        </Button>
      </div>
      <textarea
        value={csv.join(`\n`)}
        id='input-type-textarea'
        name='input-type-textarea'
        readOnly
        className={styles.csvArea}
        aria-label='Screens CSV'
      />
    </>
  );
};
