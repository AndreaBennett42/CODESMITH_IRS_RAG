import { FC } from 'react';
import styles from './AllScreens.module.scss';
import { Icon, Select } from '@trussworks/react-uswds';
import { useFlow } from '../flow/flowConfig.js';
import { useTranslation } from 'react-i18next';
import { ShowType } from './AllScreensContent.js';

type AllScreensHeaderProps = {
  expanded: boolean;
  showType: ShowType;
  onMenuToggle: () => void;
};

const AllScreensHeader: FC<AllScreensHeaderProps> = ({ onMenuToggle, expanded, showType }) => {
  const flow = useFlow();
  const { t } = useTranslation();
  const handleJump = (e: { target: { value: string } }) => {
    location.hash = `${e.target.value}`;
  };

  const handleClick = () => {
    onMenuToggle();
  };
  return (
    <div className={styles.allScreensHeader}>
      <div>
        <button className={styles.menuButton} type='button' onClick={handleClick}>
          {expanded ? (
            <>
              <Icon.Close size={4} aria-hidden='true' />
              <span className='usa-sr-only'>Close</span>
            </>
          ) : (
            <>
              <Icon.Menu size={4} aria-hidden='true' />
              <span className='usa-sr-only'>Menu</span>
            </>
          )}
        </button>
        <h1>
          <strong>Direct File</strong> | All Screens
        </h1>
      </div>
      {showType === `screens` && (
        <Select id='jump-nav' name='jump-nav' aria-label='Jump to section' onChange={handleJump}>
          <option value=''>Jump to section</option>
          {flow.categories.map((cat) => {
            const heading = t(`checklist.${cat.route}.heading`);
            return (
              <>
                <optgroup key={cat.route} label={`${heading}`}>
                  {cat.subcategories.map((subcat) => {
                    const heading = t(`checklist.${subcat.route}.heading`);
                    const headingId = heading.replace(/\s+/g, `-`).toLowerCase();
                    return (
                      <option key={subcat.route} value={headingId}>
                        {heading}
                      </option>
                    );
                  })}
                  ;
                </optgroup>
              </>
            );
          })}
        </Select>
      )}
    </div>
  );
};

export default AllScreensHeader;
