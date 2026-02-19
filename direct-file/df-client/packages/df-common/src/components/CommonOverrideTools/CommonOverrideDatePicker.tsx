import { useState, useEffect } from 'react';
import { DatePicker } from '@trussworks/react-uswds';

const getInitialDate = () => sessionStorage.getItem(`df_date_override`) || ``;

const CommonOverrideDatePicker = () => {
  const [selectedDate, setSelectedDate] = useState(getInitialDate());

  useEffect(() => {
    const storedDate = sessionStorage.getItem(`df_date_override`);
    if (storedDate) {
      setSelectedDate(storedDate);
    }
  }, []);

  const handleDateChange = (date: string | undefined) => {
    if (!date) return;
    setSelectedDate(date);
  };

  const handleConfirmDate = () => {
    const isoDate = selectedDate ? new Date(selectedDate).toISOString() : ``;
    const currentStoredDate = sessionStorage.getItem(`df_date_override`);

    if (isoDate !== currentStoredDate) {
      sessionStorage.setItem(`df_date_override`, isoDate);
      window.location.reload();
    }
  };

  const handleClearDate = () => {
    sessionStorage.removeItem(`df_date_override`);
    window.location.reload();
  };

  return (
    <div>
      <div className='display-flex'>
        <DatePicker
          id='date-override'
          name='date-override'
          defaultValue={selectedDate}
          onChange={handleDateChange}
          placeholder='MM/DD/YYYY'
        />
        <button className='usa-button' onClick={handleConfirmDate}>
          Apply
        </button>
      </div>
      <p>— or —</p>
      <button className='usa-button' onClick={handleClearDate}>
        Reset to System Date
      </button>
    </div>
  );
};

export default CommonOverrideDatePicker;
