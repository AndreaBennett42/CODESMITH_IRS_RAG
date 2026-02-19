import Translation from '../Translation/index.js';

const RequiredMarker = ({ inline = true }) => {
  // TODO: Update Truss components to use requiredMarker prop, but ONLY if
  // the title attribute can be translated
  if (inline) {
    return (
      <span aria-hidden='true' className='text-base'>
        (<Translation i18nKey='fields.generics.requiredExplainerSimple' collectionId={null} />)
      </span>
    );
  } else {
    return (
      <p data-testid='required-explainer' aria-hidden='true' className='text-base margin-bottom-0'>
        (<Translation i18nKey='fields.generics.requiredExplainerSimple' collectionId={null} />)
      </p>
    );
  }
};

export default RequiredMarker;
