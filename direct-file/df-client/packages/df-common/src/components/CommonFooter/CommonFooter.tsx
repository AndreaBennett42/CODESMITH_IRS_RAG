import {
  Identifier,
  IdentifierIdentity,
  IdentifierLink,
  IdentifierLinkItem,
  IdentifierLinks,
  IdentifierLogo,
  IdentifierLogos,
  IdentifierMasthead,
  Link,
} from '@trussworks/react-uswds';
import { CommonFooterProps } from './CommonFooterProps.types.js';
import './CommonFooter.scss';
import { FC } from 'react';
import { useTranslation } from 'react-i18next';
import { Link as ReactLink } from 'react-router-dom';

const CommonFooter: FC<CommonFooterProps> = ({
  masthead,
  df_prefix = ``,
  logo_alt,
  official_of,
  important_links,
  about,
  direct_file_news,
  accessibility,
  privacy,
}) => {
  const { i18n } = useTranslation(`translation`);

  const accessibilityUrl = i18n.language.startsWith(`es`)
    ? `https://www.irs.gov/es/accessibility`
    : `https://www.irs.gov/accessibility`;
  const privacyUrl = i18n.language.startsWith(`es`)
    ? `https://www.irs.gov/es/privacy-disclosure/irs-privacy-policy`
    : `https://www.irs.gov/privacy-disclosure/irs-privacy-policy`;
  const directFileNewsUrl = i18n.language.startsWith(`es`)
    ? `https://www.irs.gov/es/filing/irs-direct-file-for-free`
    : `https://www.irs.gov/filing/irs-direct-file-for-free`;

  return (
    <footer aria-label='direct file' className='footer'>
      <Identifier>
        <IdentifierMasthead aria-label={masthead}>
          <IdentifierLogos>
            <IdentifierLogo href='https://irs.gov'>
              <img
                src={`${df_prefix}/imgs/irs_logo_bk.svg`}
                className='usa-identifier__logo-img'
                alt={logo_alt}
                height='1em'
                width='1em'
              />
            </IdentifierLogo>
          </IdentifierLogos>
          <IdentifierIdentity domain='IRS.gov'>
            {`${official_of} `}
            <Link href='https://irs.gov'>IRS</Link>
          </IdentifierIdentity>
        </IdentifierMasthead>
        <IdentifierLinks navProps={{ 'aria-label': important_links }}>
          <IdentifierLinkItem>
            <ReactLink to={`/about/`} className='usa-link usa-identifier__required-link'>
              {about}
            </ReactLink>
          </IdentifierLinkItem>
          <IdentifierLinkItem>
            <IdentifierLink href={directFileNewsUrl}>{direct_file_news}</IdentifierLink>
          </IdentifierLinkItem>
          <IdentifierLinkItem>
            <IdentifierLink href={accessibilityUrl}>{accessibility}</IdentifierLink>
          </IdentifierLinkItem>
          <IdentifierLinkItem>
            <IdentifierLink href={privacyUrl}>{privacy}</IdentifierLink>
          </IdentifierLinkItem>
        </IdentifierLinks>
      </Identifier>
    </footer>
  );
};

export default CommonFooter;
