import classNames from 'classnames';
import { IconDisplayConfig, FlowComponentConfig } from '../flow/ContentDeclarations.js';
import { ContextHeading } from './ContextHeading/ContextHeading.js';
import Heading from './Heading.js';
import IconDisplay from './IconDisplay/IconDisplay.js';
import { conditionsAsKeySuffix } from './screenUtils.js';
import styles from './Screen.module.scss';

export const ScreenHeader = ({
  icons,
  contextHeadings,
  headings,
  collectionId,
  isDraft,
}: {
  icons: IconDisplayConfig[];
  contextHeadings: FlowComponentConfig<`ContextHeading`>[];
  headings: FlowComponentConfig<`Heading`>[];
  collectionId: string | null;
  isDraft: boolean;
}) => (
  <div className={classNames(`screen__header`, { [styles.draftContent]: isDraft })}>
    {icons.map((icon) => (
      <IconDisplay key={`${icon.props.name}-${conditionsAsKeySuffix(icon)}`} {...icon.props} />
    ))}
    {contextHeadings.map((ch) => (
      <ContextHeading
        key={`${ch.props.i18nKey}-${conditionsAsKeySuffix(ch)}`}
        {...ch.props}
        collectionId={collectionId}
      />
    ))}
    {headings.map((h) => (
      <Heading key={`${h.props.i18nKey}--${conditionsAsKeySuffix(h)}`} {...h.props} collectionId={collectionId} />
    ))}
  </div>
);
