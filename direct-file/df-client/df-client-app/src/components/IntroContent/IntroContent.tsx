import ContentDisplay from '../ContentDisplay/index.js';
import { InfoDisplayProps } from '../../types/core.js';

const IntroContent = (props: InfoDisplayProps) => {
  const allowedTags = [`p`, `ol`, `li`, `ul`];
  return <ContentDisplay {...props} allowedTags={allowedTags} />;
};

export default IntroContent;
