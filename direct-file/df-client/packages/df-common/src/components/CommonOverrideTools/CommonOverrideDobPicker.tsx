import { useState } from 'react';
import { Button, DatePicker } from '@trussworks/react-uswds';

const DF_DOB_OVERRIDE = `df_dob_override`;
const X_DATA_IMPORT_DOB = `x-data-import-dob`;
const DOB_OVERRIDE = `dob-override`;

const CommonOverrideDobPicker = () => {
  const [selectedDob, setSelectedDob] = useState(sessionStorage.getItem(DF_DOB_OVERRIDE) || ``);

  const handleDateChange = (date: string | undefined) => {
    if (!date) return;
    setSelectedDob(date);
  };

  const handleConfirmDate = () => {
    const selectedDate = new Date(selectedDob);
    const isoDate = selectedDob ? selectedDate.toISOString() : ``;
    const currentStoredDob = sessionStorage.getItem(DF_DOB_OVERRIDE);

    if (isoDate !== currentStoredDob) {
      sessionStorage.setItem(DF_DOB_OVERRIDE, isoDate);
      sessionStorage.setItem(
        X_DATA_IMPORT_DOB,
        `[` + selectedDate.getFullYear() + `,` + (selectedDate.getMonth() + 1) + `,` + selectedDate.getDate() + `]`
      );
    }
  };

  const handleClearDate = (e: React.MouseEvent<HTMLButtonElement>) => {
    sessionStorage.removeItem(DF_DOB_OVERRIDE);
    sessionStorage.removeItem(X_DATA_IMPORT_DOB);
    setSelectedDob(``);

    // setSelectedDob is async, so build in a delay before trying to clear out date input otherwise
    // it will load with the previous selectedDob as the default value
    setTimeout(() => ((document.getElementById(DOB_OVERRIDE) as HTMLInputElement).value = ``), 100);

    e.currentTarget?.blur();
  };

  return (
    <div>
      <div className='display-flex'>
        <DatePicker
          id={DOB_OVERRIDE}
          name={DOB_OVERRIDE}
          defaultValue={selectedDob}
          onChange={handleDateChange}
          onBlur={handleConfirmDate}
          placeholder='MM/DD/YYYY'
        />
        <Button type={`button`} onClick={handleClearDate}>
          Clear
        </Button>
      </div>
    </div>
  );
};

export default CommonOverrideDobPicker;
