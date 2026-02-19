import { DateInput, DateInputGroup } from '@trussworks/react-uswds';
import * as sfg from '@irs/js-factgraph-scala';
import { forwardRef, useCallback, useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { DatePickerFactProps } from '../../../types/core.js';
import useFact from '../../../hooks/useFact.js';
import { useFactControl } from '../../../hooks/useFactControl.js';
import type { DayValidationFailure, Day } from '@irs/js-factgraph-scala';
import { ComplexFormControl, buildFormControlId, buildHintId, buildHintKey } from '../../FormControl/index.js';
import Hint from '../../HelperText/Hint.js';
import Translation from '../../Translation/index.js';
import { getFactStringLimit } from '../../../factgraph/factLimitHelpers.js';
import { useFactGraph } from '../../../factgraph/FactGraphContext.js';

const IRS_CREATION_YEAR = 1862;
const BLANK_DATE_VALUE_WHEN_OPTIONAL = ``;

/* Notes on validation flow
  Dates entered by the user are validated twice. First they are validated on the client via the validateDate() 
  function in this file. Then after this client-side validation, the date is then further validated on the 
  Scala-side via the DayFactory (see factParser in this file). Most of the validation logic has been duplicated.
  Todo: remove redundant validation logic from DayFactory
   The order is as follows:
   1.  The useEffect block is the first check for validation errors when the user starts typing in any
       of the inputs.
   2.  validateDate 
     - if the date is empty, we have no errors and we can move to formatDate. 
     - if the date is incomplete or invalid, an error code is returned (but is only displayed after clicking the 
       Save and Continue button)
     - when the date is valid, we return null and we proceed to…
   3. formatDate - When we have a full day, month, and year, we can send that as 
      a string to sanitizeWhenRequired. Or if the date is optional, we send the empty string to sanitizeWhenRequired.
   4. sanitizeWhenRequired - this sends an empty string or valid date to useFactControl
*/

// format date for DayFactory
const formatDate = (date: Day, required: boolean): string => {
  const { day, month, year } = date;
  // if the date is not required but also has no values then return an empty string
  if (!required && !day && !month && !year) {
    return BLANK_DATE_VALUE_WHEN_OPTIONAL;
  } else {
    const yearStr = year ? year.toString().padStart(4, `0`) : ``;
    const monthStr = month ? month.toString().padStart(2, `0`) : ``;
    const dayStr = day ? day.toString().padStart(2, `0`) : ``;
    return `${yearStr}-${monthStr}-${dayStr}`;
  }
};

/*
 * Validates the date the user enters.
 * @param {Day} date - The date the user has entered in FactGraph format.
 * @param {Date} lastAllowableDate - optional maximum date, after which the date is invalid.
 * @param {boolean | undefined} disallowPastDates - whether a date in the past is valid.
 * @param {required | undefined} required - whether the date field is required or optional.
 * @param {boolean} exceptYear - whether to not validate the year, assuming it's not user-editable.
 * @return {string | null} the key for the validation error (if any) or null if no errors.
 */
export const validateDate = (
  date: Day,
  lastAllowableDate: Date | undefined,
  disallowPastDates: boolean | undefined,
  required: boolean | undefined,
  exceptYear?: boolean
): string | null => {
  const { day, month, year } = date;
  if (!required && day === 0 && month === 0 && year === 0) {
    return null;
  }
  if (!date) {
    return `RequiredField`;
  }

  // check if any of the fields are non-numbers, empty, or falsey:
  if (Number.isNaN(day) || Number.isNaN(month)) {
    return exceptYear ? `InvalidMDDate` : `InvalidDate`;
  } else if (Number.isNaN(year)) {
    return `InvalidDate`;
  } else if (day === 0 || month === 0 || year === 0 || !day || !month || !year) {
    return `RequiredField`;
    // this year check is a sanity check and not a technical requirement
  } else if (year < 100) {
    return `InvalidDate`;
  } else if (year < IRS_CREATION_YEAR) {
    return `InvalidYear`;
  } else {
    // Selected date should be assignable to a Date object.
    let selectedDate = undefined;
    try {
      selectedDate = new Date(month + `/` + day + `/` + year);
      if (!selectedDate || isNaN(selectedDate.getTime())) {
        return `InvalidDate`;
      }
      const isLeapYear = new Date(year, 1, 29).getMonth() === 1;
      if (!isLeapYear && month === 2 && day > 28) {
        return `InvalidDayDueToLeapYear`;
      }
    } catch (ex) {
      return `InvalidDate`;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // If last allowable date is provided, do not allow dates after the specified date.
    // Allow today's date if last allowable date is in the past. (`disallowPastDates` must be set to prevent
    // days after last allowable date and before today)
    if (
      lastAllowableDate &&
      selectedDate.getTime() > lastAllowableDate.getTime() &&
      selectedDate.getTime() > today.getTime()
    ) {
      return `AfterFilingDeadline`;
    }

    // Do not allow dates in the past.
    if (disallowPastDates && selectedDate.getTime() < today.getTime()) {
      return `PastDate`;
    }

    return null;
  }
};

export const DatePicker = forwardRef<HTMLInputElement, DatePickerFactProps>(
  (
    {
      path,
      onValidData,
      showFeedback,
      isValid,
      concretePath,
      collectionId,
      lockYearTo,
      lastAllowableDatePath,
      disallowPastDates,
      readOnly,
      hintKey,
      required = true,
      autoComplete,
    },
    ref
  ) => {
    const [fact, setFact, clearFact] = useFact<Day>(concretePath);
    const { factGraph } = useFactGraph();

    const [validationError, setValidationError] = useState(``);

    const setValidity = (isValid: boolean) => {
      onValidData(concretePath, isValid);
    };
    const factParser = useCallback(
      (day: string) => {
        const limit = getFactStringLimit(path, `Max`);
        return sfg.DayFactory(day, limit);
      },
      [path]
    );

    // This is where we send the entry to useFactControl to send to scala
    const sanitizeWhenRequired = (rawValue: string) => {
      // If we receive an empty string from formateDate and the field
      // is optional, we can submit as Blank, and it shows in the data view as "Blank"
      if (!required && !rawValue.length) {
        return { value: BLANK_DATE_VALUE_WHEN_OPTIONAL, isEmpty: true };
      } else {
        return { value: rawValue, isEmpty: false };
      }
    };
    const { onChange } = useFactControl({
      setValidity,
      setFact,
      clearFact,
      isFactRequired: required,
      onError: (failureReason: DayValidationFailure) => setValidationError(failureReason.validationMessage.toString()),
      factParser,
      sanitize: sanitizeWhenRequired,
    });

    // Destructure sfg date to set default values on UI elements:
    const { day: sfgDay, month: sfgMonth, year: sfgYear } = fact || {};

    const lockYear = useMemo(() => {
      if (lockYearTo) {
        if (factGraph.get(lockYearTo).complete) {
          return Number(factGraph.get(lockYearTo).get.toString());
        } else {
          throw new Error(`DatePicker is trying to lock the year to fact ${lockYearTo} which is not complete`);
        }
      } else {
        return null;
      }
    }, [lockYearTo, factGraph]);

    const lastAllowableDate = useMemo(() => {
      if (lastAllowableDatePath) {
        if (factGraph.get(lastAllowableDatePath).hasValue) {
          const date = new Date(factGraph.get(lastAllowableDatePath).get.toString());

          // Adding a day because the date specified should be allowed.
          date.setDate(date.getDate() + 1);
          date.setHours(0, 0, 0, 0);
          return date;
        } else {
          throw new Error(
            `DatePicker is trying to set the max date to fact ${lastAllowableDatePath} which is not complete`
          );
        }
      } else {
        return undefined;
      }
    }, [lastAllowableDatePath, factGraph]);

    // state variables to hold the user entered date. initialized to sfg data
    // unless sfg data is blank.
    const [date, setDate] = useState<Day>({
      day: sfgDay || 0,
      month: sfgMonth || 0,
      year: lockYear || sfgYear || 0,
    });

    const { t } = useTranslation(`translation`);

    const onFieldChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value } = e.target;

      setDate((prevDate) => ({
        ...prevDate,
        [name]: Number(value),
      }));
    }, []);

    useEffect(() => {
      const exceptYear = null !== lockYear;
      const dateValidationError = validateDate(date, lastAllowableDate, disallowPastDates, required, exceptYear);
      if (dateValidationError) {
        setValidationError(dateValidationError);
        setValidity(false);
      } else {
        onChange(formatDate(date, required));
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [date, required]);

    const showError = showFeedback && !isValid;
    const hintId = buildHintId(buildFormControlId(concretePath));

    const errorMessage =
      validationError !== `` ? (
        <Translation
          i18nKey={[
            `fields.${path}.errorMessages.${validationError}`,
            `fields.generics.date.errorMessages.${validationError}`,
          ]}
          collectionId={collectionId}
        />
      ) : (
        <></>
      );

    return (
      <ComplexFormControl
        path={path}
        concretePath={concretePath}
        showError={showError}
        errorMessage={errorMessage}
        required={required}
        labelledBy='legend'
        hintId={hintId}
      >
        <Hint hintId={hintId} i18nKey={hintKey || buildHintKey(path)} collectionId={collectionId} />
        <DateInputGroup id={`id-${concretePath}`}>
          <DateInput
            id='month'
            inputRef={ref}
            name='month'
            label={t(`datePicker.month`)}
            unit='month'
            inputMode='numeric'
            defaultValue={sfgMonth ?? ``}
            onChange={onFieldChange}
            maxLength={2}
            minLength={1}
            required={required}
            disabled={readOnly}
            autoComplete={autoComplete && `${autoComplete}-month`}
          />
          <DateInput
            id='day'
            name='day'
            label={t(`datePicker.day`)}
            unit='day'
            inputMode='numeric'
            defaultValue={sfgDay ?? ``}
            onChange={onFieldChange}
            maxLength={2}
            minLength={1}
            required={required}
            disabled={readOnly}
            autoComplete={autoComplete && `${autoComplete}-day`}
          />
          {!lockYearTo && (
            <DateInput
              id='year'
              name='year'
              label={t(`datePicker.year`)}
              unit='year'
              inputMode='numeric'
              defaultValue={sfgYear ?? ``}
              onChange={onFieldChange}
              maxLength={4}
              minLength={4}
              required={required}
              disabled={readOnly}
              autoComplete={autoComplete && `${autoComplete}-year`}
            />
          )}
        </DateInputGroup>
      </ComplexFormControl>
    );
  }
);

DatePicker.displayName = `DatePicker`;

export default DatePicker;
