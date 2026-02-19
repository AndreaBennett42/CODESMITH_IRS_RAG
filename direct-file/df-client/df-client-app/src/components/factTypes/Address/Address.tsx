import * as sfg from '@irs/js-factgraph-scala';
import { AddressValidationFailure } from '@irs/js-factgraph-scala';
import { CharacterCount, Select, TextInput } from '@trussworks/react-uswds';
import { ReactElement, forwardRef, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import useFact from '../../../hooks/useFact.js';
import { useFactControl } from '../../../hooks/useFactControl.js';
import { SelectOption, AddressFactProps } from '../../../types/core.js';
import { ComplexFormControl, FormFieldWrapper, buildFormControlId, sanitizeString } from '../../FormControl/index.js';
import { QuietErrorMessage } from '../../QuietErrorMessage.js';
import DFModal from '../../HelperText/DFModal.js';
import { StateOrProvince } from '../../../types/StateOrProvince.js';
import ContentDisplay from '../../ContentDisplay/index.js';

const BOUNDED_NEW_LINE_CHARS_PATTERN = /^(\r?\n|\r|\n)+|(\r?\n|\r|\n)+$/g;
const STREET_MAX_LENGTH = 35;
const CITY_MAX_LENGTH = 22;

export const getStreetCharacterCount = (text: string): number => {
  const trimmedText = text.replace(BOUNDED_NEW_LINE_CHARS_PATTERN, ``);

  const replaceNewLineCharsWithSpace = trimmedText.replace(/(\r?\n|\r|\n)+/g, ` `);

  return replaceNewLineCharsWithSpace.length;
};

const FIELDS_TO_SANITIZE = [`streetAddress`, `streetAddressLine2`, `city`];

const Address = forwardRef<HTMLInputElement, AddressFactProps>(
  (
    {
      path,
      onValidData,
      showFeedback,
      isValid,
      concretePath,
      collectionId,
      useCombinedStreetLengthForValidation,
      readOnly,
      hintKey,
      autoComplete,
    },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<sfg.Address>(concretePath);
    const [validationErrors, setValidationErrors] = useState(new Map());

    const setValidity = (isValid: boolean) => {
      onValidData(concretePath, isValid);
    };

    const { onChange } = useFactControl({
      setValidity,
      setFact,
      clearFact,
      isFactRequired: true, // Optional not implemented for Address
      onError: (failureReason: AddressValidationFailure) => {
        setValidationErrors(sfg.scalaMapToJsMap(failureReason.addressErrors));
        onValidData(concretePath, false);
      },
      factParser: (address: sfg.Address) =>
        sfg.AddressFactory(
          address.streetAddress,
          address.city,
          address.postalCode,
          address.stateOrProvence,
          address.streetAddressLine2,
          address.country
        ),
    });

    // Destructure the fact. This data doesn't persist, unless the button is clicked
    // and the screen has all valid fields
    const {
      streetAddress: sfgStreetL1 = ``,
      city: sfgCity = ``,
      postalCode: sfgZip = ``,
      stateOrProvence: sfgState = ``,
      streetAddressLine2: sfgStreetL2 = ``,
      country: sfgCountry = ``,
    } = fact || {};

    // Store the fact in state to allow persistence during error validation
    const [address, setAddress] = useState({
      streetAddress: sfgStreetL1 ?? ``,
      city: sfgCity ?? ``,
      postalCode: sfgZip ?? ``,
      stateOrProvence: sfgState ?? ``,
      streetAddressLine2: sfgStreetL2 ?? ``,
      country: sfgCountry ?? ``,
    });

    const { t } = useTranslation(`translation`);

    const STATE_OR_PROVINCE_OPTIONS: SelectOption<StateOrProvince>[] = [
      { value: `AL`, label: `${t(`enums.statesAndProvinces.AL`)}` },
      { value: `AK`, label: `${t(`enums.statesAndProvinces.AK`)}` },
      { value: `AZ`, label: `${t(`enums.statesAndProvinces.AZ`)}` },
      { value: `AR`, label: `${t(`enums.statesAndProvinces.AR`)}` },
      { value: `CA`, label: `${t(`enums.statesAndProvinces.CA`)}` },
      { value: `CO`, label: `${t(`enums.statesAndProvinces.CO`)}` },
      { value: `CT`, label: `${t(`enums.statesAndProvinces.CT`)}` },
      { value: `DE`, label: `${t(`enums.statesAndProvinces.DE`)}` },
      { value: `DC`, label: `${t(`enums.statesAndProvinces.DC`)}` },
      { value: `FL`, label: `${t(`enums.statesAndProvinces.FL`)}` },
      { value: `GA`, label: `${t(`enums.statesAndProvinces.GA`)}` },
      { value: `HI`, label: `${t(`enums.statesAndProvinces.HI`)}` },
      { value: `ID`, label: `${t(`enums.statesAndProvinces.ID`)}` },
      { value: `IL`, label: `${t(`enums.statesAndProvinces.IL`)}` },
      { value: `IN`, label: `${t(`enums.statesAndProvinces.IN`)}` },
      { value: `IA`, label: `${t(`enums.statesAndProvinces.IA`)}` },
      { value: `KS`, label: `${t(`enums.statesAndProvinces.KS`)}` },
      { value: `KY`, label: `${t(`enums.statesAndProvinces.KY`)}` },
      { value: `LA`, label: `${t(`enums.statesAndProvinces.LA`)}` },
      { value: `ME`, label: `${t(`enums.statesAndProvinces.ME`)}` },
      { value: `MD`, label: `${t(`enums.statesAndProvinces.MD`)}` },
      { value: `MA`, label: `${t(`enums.statesAndProvinces.MA`)}` },
      { value: `MI`, label: `${t(`enums.statesAndProvinces.MI`)}` },
      { value: `MN`, label: `${t(`enums.statesAndProvinces.MN`)}` },
      { value: `MS`, label: `${t(`enums.statesAndProvinces.MS`)}` },
      { value: `MO`, label: `${t(`enums.statesAndProvinces.MO`)}` },
      { value: `MT`, label: `${t(`enums.statesAndProvinces.MT`)}` },
      { value: `NE`, label: `${t(`enums.statesAndProvinces.NE`)}` },
      { value: `NV`, label: `${t(`enums.statesAndProvinces.NV`)}` },
      { value: `NH`, label: `${t(`enums.statesAndProvinces.NH`)}` },
      { value: `NJ`, label: `${t(`enums.statesAndProvinces.NJ`)}` },
      { value: `NM`, label: `${t(`enums.statesAndProvinces.NM`)}` },
      { value: `NY`, label: `${t(`enums.statesAndProvinces.NY`)}` },
      { value: `NC`, label: `${t(`enums.statesAndProvinces.NC`)}` },
      { value: `ND`, label: `${t(`enums.statesAndProvinces.ND`)}` },
      { value: `OH`, label: `${t(`enums.statesAndProvinces.OH`)}` },
      { value: `OK`, label: `${t(`enums.statesAndProvinces.OK`)}` },
      { value: `OR`, label: `${t(`enums.statesAndProvinces.OR`)}` },
      { value: `PA`, label: `${t(`enums.statesAndProvinces.PA`)}` },
      { value: `RI`, label: `${t(`enums.statesAndProvinces.RI`)}` },
      { value: `SC`, label: `${t(`enums.statesAndProvinces.SC`)}` },
      { value: `SD`, label: `${t(`enums.statesAndProvinces.SD`)}` },
      { value: `TN`, label: `${t(`enums.statesAndProvinces.TN`)}` },
      { value: `TX`, label: `${t(`enums.statesAndProvinces.TX`)}` },
      { value: `UT`, label: `${t(`enums.statesAndProvinces.UT`)}` },
      { value: `VT`, label: `${t(`enums.statesAndProvinces.VT`)}` },
      { value: `VA`, label: `${t(`enums.statesAndProvinces.VA`)}` },
      { value: `WA`, label: `${t(`enums.statesAndProvinces.WA`)}` },
      { value: `WV`, label: `${t(`enums.statesAndProvinces.WV`)}` },
      { value: `WI`, label: `${t(`enums.statesAndProvinces.WI`)}` },
      { value: `WY`, label: `${t(`enums.statesAndProvinces.WY`)}` },
      { value: `AA`, label: `${t(`enums.armedForces.armedForcesAmericas`)}` },
      { value: `AE`, label: `${t(`enums.armedForces.armedForcesAfrica`)}` },
      { value: `AE`, label: `${t(`enums.armedForces.armedForcesCanada`)}` },
      { value: `AE`, label: `${t(`enums.armedForces.armedForcesEurope`)}` },
      { value: `AE`, label: `${t(`enums.armedForces.armedForcesMiddleEast`)}` },
      { value: `AP`, label: `${t(`enums.armedForces.armedForcesPacific`)}` },
    ];

    const onFieldChange = (e: React.ChangeEvent<HTMLInputElement> | React.ChangeEvent<HTMLSelectElement>) => {
      const { name, value } = e.target;

      if (useCombinedStreetLengthForValidation && name === `streetAddress`) {
        // trim any leading or lagging newline characters
        const trimValue = value.replace(BOUNDED_NEW_LINE_CHARS_PATTERN, ``);

        // split trimmed value on each newline char
        const [addressLine1, ...addressRemaining] = trimValue ? trimValue.split(/\r?\n|\r|\n/g) : [``];
        const streetAddress = sanitizeString(addressLine1).value;

        // Note: While the UI only shows a row text area, the user could enter any number of newlines
        // greater than 2. This will wrap all lines after the 2nd line into the 2nd line.
        let streetAddressLine2 = addressRemaining.some((str) => str.length > 0) ? addressRemaining.join(` `) : ``;
        streetAddressLine2 = sanitizeString(streetAddressLine2).value;

        setAddress((prevAddress) => ({
          ...prevAddress,
          streetAddress: streetAddress,
          streetAddressLine2: streetAddressLine2,
        }));
      } else {
        setAddress((prevAddress) => ({
          ...prevAddress,
          [name]: FIELDS_TO_SANITIZE.includes(name) ? sanitizeString(value).value : value,
        }));
      }
    };

    useEffect(() => {
      onChange(address);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [address]);

    const showError = showFeedback && !isValid;

    interface InlineFeedback {
      streetAddress?: ReactElement;
      streetAddressLine2?: ReactElement;
      city?: ReactElement;
      postalCode?: ReactElement;
      stateOrProvence?: ReactElement;
    }
    const inlineFeedback: InlineFeedback = {};

    const controlId = buildFormControlId(concretePath);
    function buildErrorMessageId(field: keyof InlineFeedback) {
      return `${controlId}__${field}__error-msg` as const;
    }
    function buildFieldControlId(field: keyof sfg.Address | ``) {
      return `${controlId}__${field}` as const;
    }

    // TODO expand this if/when we actually have more cases to handle (e.g. mailing vs residence, spouse, sections, etc)
    function buildAutoComplete(field: string) {
      if (!autoComplete) return undefined;
      if (autoComplete === `street-address`) {
        return field;
      } else {
        return `${autoComplete} ${field}`;
      }
    }

    const buildI18nKeys = useCallback(
      (suffix: string) => {
        return [`fields.${path}.${suffix}`, `fields.generics.address.${suffix}`];
      },
      [path]
    );

    // ERRORS
    const getCountMessage = (count: number, maxCount: number): string => {
      const remaining = maxCount - count;
      if (remaining >= 0) {
        return t(buildI18nKeys(`errorMessages.charactersLeft`), { count: remaining });
      } else {
        return t(buildI18nKeys(`errorMessages.charactersOverLimit`), { count: -remaining });
      }
    };
    if (showError) {
      for (const field of validationErrors.keys() as Iterable<keyof InlineFeedback>) {
        inlineFeedback[field] = (
          <QuietErrorMessage id={buildErrorMessageId(field)}>
            {t(buildI18nKeys(`errorMessages.${validationErrors.get(`${field}`).validationMessage.toString()}`))}
          </QuietErrorMessage>
        );
      }
    }

    const showHelpTextForStreet =
      showError &&
      [`InvalidStreetLength`, `InvalidStreetLine2Length`, `InvalidTotalStreetChars`].includes(
        validationErrors.get(`streetAddress`)?.validationMessage.toString()
      );

    return (
      <ComplexFormControl
        path={path}
        concretePath={concretePath}
        showError={showError}
        errorMessage={t(`enums.messages.complexFactError`, { count: validationErrors.size })}
        labelledBy='self'
      >
        <FormFieldWrapper
          labelKey={buildI18nKeys(`country`)}
          controlId={buildFieldControlId(`country`)}
          collectionId={collectionId}
          showError={false}
        >
          <DFModal i18nKey={hintKey ?? ``} collectionId={collectionId} />
          <Select
            id={buildFieldControlId(`country`)}
            disabled
            name='country'
            defaultValue={sfgCountry ?? ``}
            // autoComplete={autoComplete === `street-address` ? `country-name` : undefined}
            autoComplete={buildAutoComplete(`country-name`)}
          >
            <option value='USA'>{t(buildI18nKeys(`unitedStatesOfAmerica`))}</option>
          </Select>
        </FormFieldWrapper>

        {useCombinedStreetLengthForValidation ? (
          <FormFieldWrapper
            labelKey={buildI18nKeys(`streetAddress`)}
            controlId={buildFieldControlId(`streetAddress`)}
            collectionId={collectionId}
            showError={
              showError && (validationErrors.has(`streetAddress`) || validationErrors.has(`streetAddressLine2`))
            }
            errorMessage={
              inlineFeedback.streetAddress ? (
                inlineFeedback.streetAddress
              ) : inlineFeedback.streetAddressLine2 ? (
                inlineFeedback.streetAddressLine2
              ) : (
                <></>
              )
            }
          >
            <CharacterCount
              inputRef={ref}
              id={buildFieldControlId(`streetAddress`)}
              type='text'
              name='streetAddress'
              defaultValue={sfgStreetL2 ? `${sfgStreetL1}\n${sfgStreetL2}` : sfgStreetL1}
              onChange={onFieldChange}
              aria-invalid={
                showError && (validationErrors.has(`streetAddress`) || validationErrors.has(`streetAddressLine2`))
              }
              error={showError && (validationErrors.has(`streetAddress`) || validationErrors.has(`streetAddressLine2`))}
              aria-describedby={
                showError && (validationErrors.has(`streetAddress`) || validationErrors.has(`streetAddressLine2`))
                  ? buildErrorMessageId(`streetAddress`)
                  : undefined
              }
              maxLength={STREET_MAX_LENGTH}
              isTextArea={true}
              rows={2}
              getCharacterCount={getStreetCharacterCount}
              getMessage={getCountMessage}
              required
              disabled={readOnly}
              autoComplete={buildAutoComplete(`street-address`)}
            />
          </FormFieldWrapper>
        ) : (
          <>
            <FormFieldWrapper
              labelKey={buildI18nKeys(`streetAddress`)}
              controlId={buildFieldControlId(`streetAddress`)}
              collectionId={collectionId}
              showError={showError && validationErrors.has(`streetAddress`)}
              errorMessage={inlineFeedback.streetAddress ? inlineFeedback.streetAddress : <></>}
            >
              <CharacterCount
                inputRef={ref}
                id={buildFieldControlId(`streetAddress`)}
                type='text'
                name='streetAddress'
                defaultValue={sfgStreetL1}
                onChange={onFieldChange}
                aria-invalid={showError && validationErrors.has(`streetAddress`)}
                error={showError && validationErrors.has(`streetAddress`)}
                aria-describedby={
                  showError && validationErrors.has(`streetAddress`) ? buildErrorMessageId(`streetAddress`) : undefined
                }
                maxLength={STREET_MAX_LENGTH}
                isTextArea={false}
                getCharacterCount={getStreetCharacterCount}
                getMessage={getCountMessage}
                required
                disabled={readOnly}
                autoComplete={buildAutoComplete(`address-line1`)}
              />
            </FormFieldWrapper>
            <FormFieldWrapper
              labelKey={buildI18nKeys(`streetAddressLine2`)}
              controlId={buildFieldControlId(`streetAddressLine2`)}
              collectionId={collectionId}
              showError={showError && validationErrors.has(`streetAddressLine2`)}
              errorMessage={inlineFeedback.streetAddressLine2 ? inlineFeedback.streetAddressLine2 : <></>}
              required={false}
            >
              <CharacterCount
                inputRef={ref}
                id={buildFieldControlId(`streetAddressLine2`)}
                type='text'
                name='streetAddressLine2'
                defaultValue={sfgStreetL2}
                onChange={onFieldChange}
                aria-invalid={showError && validationErrors.has(`streetAddressLine2`)}
                error={showError && validationErrors.has(`streetAddressLine2`)}
                aria-describedby={
                  showError && validationErrors.has(`streetAddressLine2`)
                    ? buildErrorMessageId(`streetAddressLine2`)
                    : undefined
                }
                maxLength={STREET_MAX_LENGTH}
                getCharacterCount={getStreetCharacterCount}
                getMessage={getCountMessage}
                disabled={readOnly}
                autoComplete={buildAutoComplete(`address-line2`)}
              />
            </FormFieldWrapper>
          </>
        )}

        {showHelpTextForStreet ? (
          // TODO ContentDisplay doesn't allow for a string array for fallbacks
          <ContentDisplay i18nKey={`fields.generics.address.errorMessages.InvalidStreetCharsDetail`} />
        ) : (
          <></>
        )}

        <FormFieldWrapper
          labelKey={buildI18nKeys(`city`)}
          controlId={buildFieldControlId(`city`)}
          collectionId={collectionId}
          showError={showError && validationErrors.has(`city`)}
          errorMessage={inlineFeedback.city ? inlineFeedback.city : <></>}
        >
          <CharacterCount
            id={buildFieldControlId(`city`)}
            type='text'
            name='city'
            defaultValue={sfgCity ?? ``}
            onChange={onFieldChange}
            aria-invalid={showError && validationErrors.has(`city`)}
            validationStatus={showError && validationErrors.has(`city`) ? `error` : undefined}
            maxLength={CITY_MAX_LENGTH}
            getMessage={getCountMessage}
            required
            aria-describedby={showError && validationErrors.has(`city`) ? buildErrorMessageId(`city`) : undefined}
            disabled={readOnly}
            autoComplete={buildAutoComplete(`address-level2`)}
          />
        </FormFieldWrapper>

        <FormFieldWrapper
          labelKey={buildI18nKeys(`stateOrProvence`)}
          controlId={buildFieldControlId(`stateOrProvence`)}
          collectionId={collectionId}
          showError={showError && validationErrors.has(`stateOrProvence`)}
          errorMessage={inlineFeedback.stateOrProvence ? inlineFeedback.stateOrProvence : <></>}
        >
          <Select
            id={buildFieldControlId(`stateOrProvence`)}
            name='stateOrProvence'
            defaultValue={sfgState ?? ``}
            onChange={onFieldChange}
            aria-invalid={showError && validationErrors.has(`stateOrProvence`)}
            validationStatus={showError && validationErrors.has(`stateOrProvence`) ? `error` : undefined}
            aria-describedby={
              showError && validationErrors.has(`stateOrProvence`) ? buildErrorMessageId(`stateOrProvence`) : undefined
            }
            required
            disabled={readOnly}
            autoComplete={buildAutoComplete(`address-level1`)}
          >
            <option value=''> {t(`select.select`)}</option>
            {STATE_OR_PROVINCE_OPTIONS.map((option) => (
              <option key={`${option.value}.${option.label}`} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </FormFieldWrapper>

        <FormFieldWrapper
          labelKey={buildI18nKeys(`postalCode`)}
          controlId={buildFieldControlId(`postalCode`)}
          collectionId={collectionId}
          showError={showError && validationErrors.has(`postalCode`)}
          errorMessage={inlineFeedback.postalCode ? inlineFeedback.postalCode : <></>}
        >
          <TextInput
            id={buildFieldControlId(`postalCode`)}
            type='text'
            name='postalCode'
            defaultValue={sfgZip ?? ``}
            onChange={onFieldChange}
            aria-invalid={showError && validationErrors.has(`postalCode`)}
            validationStatus={showError && validationErrors.has(`postalCode`) ? `error` : undefined}
            aria-describedby={
              showError && validationErrors.has(`postalCode`) ? buildErrorMessageId(`postalCode`) : undefined
            }
            required
            disabled={readOnly}
            autoComplete={buildAutoComplete(`postal-code`)}
          />
        </FormFieldWrapper>
      </ComplexFormControl>
    );
  }
);

Address.displayName = `Address`;

export default Address;
