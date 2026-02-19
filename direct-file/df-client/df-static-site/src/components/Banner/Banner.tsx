import { GovBanner } from '@trussworks/react-uswds';

// TODO: The react-uswds GovBanner  only supports English and Spanish.
// Refactor to support multiple languages using the compositional components found here:
// https://github.com/trussworks/react-uswds/tree/main/src/components/banner
type BannerProps = {
  lang: string;
};

const Banner = ({ lang }: BannerProps) => {
  return (
    <div className='bg-ink'>
      <GovBanner className='gov-banner' language={lang === `es-US` ? `spanish` : `english`} />
    </div>
  );
};

export default Banner;
