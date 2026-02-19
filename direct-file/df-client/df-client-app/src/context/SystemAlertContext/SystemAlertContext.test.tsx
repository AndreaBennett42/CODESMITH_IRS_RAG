import { act, renderHook } from '@testing-library/react';
import { store } from '../../redux/store.js';
import {
  StoredSystemAlertConfig,
  SystemAlertConfig,
  SystemAlertContextProvider,
  SystemAlertKey,
  useSystemAlertContext,
} from './SystemAlertContext.js';
import { JSXElementConstructor, ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';

const KEY = `key` as SystemAlertKey;
const CONFIG: SystemAlertConfig = {
  type: `error`,
  i18nKey: `someI18nKey`,
};
const STORED_CONFIG: StoredSystemAlertConfig = {
  alertConfig: {
    ...CONFIG,
    i18nKey: `systemAlerts.someI18nKey`,
  },
  shouldClearOnRouteChange: true,
  timestamp: 1732310988770,
};

const OTHER_KEY = `otherKey` as SystemAlertKey;
const OTHER_CONFIG: SystemAlertConfig = {
  type: `error`,
  i18nKey: `otherI18nKey`,
};
const STORED_OTHER_CONFIG: StoredSystemAlertConfig = {
  alertConfig: {
    ...OTHER_CONFIG,
    i18nKey: `systemAlerts.otherI18nKey`,
  },
  shouldClearOnRouteChange: true,
  timestamp: 1732310988784,
};

describe(`SystemAlertContext`, () => {
  const wrapper: JSXElementConstructor<{ children: ReactNode }> = ({ children }) => (
    <BrowserRouter>
      <Provider store={store}>
        <SystemAlertContextProvider>{children}</SystemAlertContextProvider>
      </Provider>
    </BrowserRouter>
  );

  const renderUseSystemAlertContext = () =>
    renderHook(() => useSystemAlertContext(), {
      wrapper,
    });

  it(`initializes with empty systemAlerts`, () => {
    const { result } = renderUseSystemAlertContext();

    expect(Object.keys(result.current.systemAlerts).length).toEqual(0);
  });

  describe(`setSystemAlert`, () => {
    it(`adds a new alert to an empty systemAlerts`, () => {
      const { result } = renderUseSystemAlertContext();
      expect(Object.keys(result.current.systemAlerts).length).toEqual(0);
      act(() => {
        result.current.setSystemAlert(KEY, CONFIG);
      });
      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);

      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(expect.objectContaining(STORED_CONFIG.alertConfig));
    });
    it(`can add multiple a new alerts to systemAlerts`, () => {
      const { result } = renderUseSystemAlertContext();
      act(() => {
        result.current.setSystemAlert(KEY, CONFIG);
      });
      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);
      act(() => {
        result.current.setSystemAlert(OTHER_KEY, OTHER_CONFIG);
      });
      expect(Object.keys(result.current.systemAlerts).length).toEqual(2);
      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(expect.objectContaining(STORED_CONFIG.alertConfig));
      expect(result.current.systemAlerts[OTHER_KEY]?.alertConfig).toEqual(
        expect.objectContaining(STORED_OTHER_CONFIG.alertConfig)
      );
    });
    it(`overwrites an existing alert within systemAlerts if the key matches`, () => {
      const { result } = renderUseSystemAlertContext();
      act(() => {
        result.current.setSystemAlert(KEY, CONFIG);
      });
      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);
      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(expect.objectContaining(STORED_CONFIG.alertConfig));
      act(() => {
        result.current.setSystemAlert(KEY, OTHER_CONFIG);
      });
      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);
      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(
        expect.objectContaining(STORED_OTHER_CONFIG.alertConfig)
      );
      //expect(result.current.systemAlerts[OTHER_KEY]).toBeUndefined();
    });
  });

  describe(`deleteSystemAlert`, () => {
    it(`does nothing if there are no system alerts`, () => {
      const { result } = renderUseSystemAlertContext();

      // Not wrapped in act, because should not trigger a state update/render if no action is taken
      // TODO: When able to return a flag, verify that the value is falsy
      result.current.deleteSystemAlert(KEY);

      expect(Object.keys(result.current.systemAlerts).length).toEqual(0);
    });

    it(`removes system alert if present`, () => {
      const { result } = renderUseSystemAlertContext();

      act(() => {
        result.current.setSystemAlert(KEY, CONFIG);
      });

      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);
      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(expect.objectContaining(STORED_CONFIG.alertConfig));

      act(() => {
        result.current.deleteSystemAlert(KEY);
      });

      expect(result.current.systemAlerts[KEY]).toBeFalsy();
      expect(Object.keys(result.current.systemAlerts).length).toEqual(0);
    });

    it(`removes only the specified system alert if present`, () => {
      const { result } = renderUseSystemAlertContext();

      act(() => {
        result.current.setSystemAlert(KEY, CONFIG);
        result.current.setSystemAlert(OTHER_KEY, OTHER_CONFIG);
      });

      expect(Object.keys(result.current.systemAlerts).length).toEqual(2);
      expect(result.current.systemAlerts[KEY]?.alertConfig).toEqual(expect.objectContaining(STORED_CONFIG.alertConfig));
      expect(result.current.systemAlerts[OTHER_KEY]?.alertConfig).toEqual(
        expect.objectContaining(STORED_OTHER_CONFIG.alertConfig)
      );

      act(() => {
        result.current.deleteSystemAlert(KEY);
      });

      expect(result.current.systemAlerts[KEY]).toBeFalsy();
      expect(result.current.systemAlerts[OTHER_KEY]).toBeTruthy();
      expect(Object.keys(result.current.systemAlerts).length).toEqual(1);
    });
  });
});
