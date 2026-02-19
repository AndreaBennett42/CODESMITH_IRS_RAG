import { useRef, RefObject } from 'react';

/**
 * 
 Use like:
    const [inputRef, setInputFocus] = useFocus()
    <button onClick={setInputFocus} >Focus</button>
    <input ref={inputRef} />
 */
export const useFocus = (): [RefObject<HTMLInputElement>, () => void] => {
  const inputRef = useRef<HTMLInputElement>(null);
  const setFocus = () => {
    if (inputRef?.current) {
      inputRef.current.focus();
      // select whole field? Could be an optional prop.
      inputRef.current.selectionStart = 0;
      inputRef.current.selectionEnd = inputRef.current.value?.length || 0;
    }
  };

  return [inputRef, setFocus];
};
