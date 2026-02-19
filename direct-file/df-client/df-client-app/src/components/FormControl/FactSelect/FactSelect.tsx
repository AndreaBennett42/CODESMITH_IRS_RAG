import {
  ChangeEventHandler,
  forwardRef,
  useCallback,
  useEffect,
  useState,
  useId,
  useRef,
  useMemo,
  createRef,
  MutableRefObject,
  LegacyRef,
} from 'react';
import * as sfg from '@irs/js-factgraph-scala';
import { useTranslation } from 'react-i18next';
import { DollarImplementation } from '../../factTypes/Dollar.js';
import { FactProps } from '../../../types/core.js';
import { getEnumOptions } from '../../../hooks/useEnumOptions.js';
import { useFactGraph } from '../../../factgraph/FactGraphContext.js';
import Translation from '../../Translation/index.js';
import { Select, Button, Fieldset, FormGroup } from '@trussworks/react-uswds';
import { Path } from '../../../flow/Path.js';
import { AbsolutePath, Path as FDPath } from '../../../fact-dictionary/Path.js';
import { FormFieldWrapper } from '../FormFieldWrapper.js';

import styles from './FactSelect.module.scss';
import { useDynamicFact } from '../../../hooks/useFact.js';
import { FACT_PATH_TYPES } from '../../../factgraph/factGraphTypeHelpers.js';
import { setSmoothScroll } from '../../../misc/misc.js';

export function getCurrentFactSelectValues(path: FDPath, collectionId: string | null, factGraph: sfg.FactGraph) {
  const { optionsPath, values: enumOptionValues } = getEnumOptions(factGraph, path, collectionId);
  const pathPrefix = path.toString().split(`/`).slice(0, -1).join(`/`) as AbsolutePath;
  const concretePathPrefix = Path.concretePath(pathPrefix, collectionId);

  if (!enumOptionValues) {
    throw new Error(`Received invalid enum options from fact graph`);
  }

  const currentValues = enumOptionValues.map((optionValue) => {
    const currentResult = factGraph.get(`${concretePathPrefix}/${optionValue}` as sfg.ConcretePath);
    return { optionValue, currentResult };
  });
  return { optionsPath, currentValues, pathPrefix, enumOptionValues };
}

function selectedFactIsBoolean(
  path: AbsolutePath | undefined,
  fact: boolean | sfg.Dollar | undefined
): fact is boolean | undefined {
  const typeName: string | undefined = path && FACT_PATH_TYPES.get(path);

  if (typeName === `Boolean`) return true;
  else if (typeName === `Dollar` || path === undefined) return false;
  else throw new TypeError(`Type of ${path} is unsupported.  Expected Boolean or Dollar, found ${typeName}`);
}

export interface BoxFactProps extends Omit<FactProps, `path` | `concretePath` | `isValid`> {
  parentPath: AbsolutePath;
  concreteParentPath: sfg.ConcretePath;
  pathPrefix: string;
  optionsPath: AbsolutePath;
  codeOptions: string[];
  selectedCodeOptions: string[];
  codeOptionValue: string | undefined;
  setCodeOptionValue: (codeOptionValue: string) => void;
  codeSelectRef: MutableRefObject<HTMLSelectElement>;
  index: number;
  shouldAllowInitialFocus: boolean;
  onRemove: () => void;
}
export function BoxFact({
  parentPath,
  concreteParentPath,
  pathPrefix,
  optionsPath,
  codeOptions,
  selectedCodeOptions,
  codeOptionValue,
  setCodeOptionValue,
  showFeedback,
  collectionId,
  onValidData,
  onRemove,
  factValidity,
  factRefs,
  handleFactRef,
  codeSelectRef,
  index,
  readOnly,
  shouldAllowInitialFocus,
  ...props
}: BoxFactProps) {
  const { t } = useTranslation();

  const [selectedPath, setSelectedPath] = useState(
    codeOptionValue ? (`${pathPrefix}/${codeOptionValue}` as AbsolutePath) : undefined
  );
  const [selectedConcretePath, setSelectedConcretePath] = useState(
    selectedPath && Path.concretePath(selectedPath, collectionId)
  );

  const [fact, setFact, clearFact, isComplete] = useDynamicFact<sfg.Dollar | boolean>(selectedConcretePath);
  const [amountIsValid, setAmountIsValid] = useState(isComplete);

  const isBoolean = selectedFactIsBoolean(selectedPath, fact);

  useEffect(() => {
    if (isBoolean) {
      // If a fact is boolean type, then being selected makes it true.
      if (fact !== true) {
        setFact(true);
        onValidData(selectedConcretePath as sfg.ConcretePath, true);
      }
    } else if (typeof fact === `boolean`) {
      // We need to clear the fact if we're switching from a boolean fact to a non-boolean fact
      clearFact();
    }
  }, [clearFact, fact, isBoolean, onValidData, selectedConcretePath, setFact]);

  const codeAmountRef = useMemo(() => {
    if (!handleFactRef) throw new Error(`handleFactRef not forwarded to BoxFact`);

    return selectedConcretePath ? handleFactRef(selectedConcretePath) : createRef<HTMLInputElement>();
  }, [handleFactRef, selectedConcretePath]);

  const setInitialFocus = useCallback(
    // Claim focus if we're a fresh, blank control
    (node: HTMLSelectElement | null) => {
      if (node !== null) {
        codeSelectRef.current = node;

        if (selectedConcretePath === undefined && shouldAllowInitialFocus) {
          node.focus();
          node.scrollIntoView({ block: `center` });
        }
      }
    },
    [codeSelectRef, selectedConcretePath, shouldAllowInitialFocus]
  );

  const innerOnValidData = useCallback(
    (path: sfg.ConcretePath, validity: boolean) => {
      setAmountIsValid(validity);

      if (selectedConcretePath !== undefined) {
        onValidData(path, validity);
      }
    },
    [onValidData, selectedConcretePath]
  );

  const onPathChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      const newOptionValue = e.target.value;
      const newPath = `${pathPrefix}/${newOptionValue}` as AbsolutePath;
      if (newPath !== selectedPath) {
        if (selectedConcretePath) {
          // Clear validity and control metadata
          factValidity?.delete(selectedConcretePath);
          factRefs?.current.delete(selectedConcretePath);
        }

        const newConcretePath = Path.concretePath(newPath, collectionId);

        setCodeOptionValue(newOptionValue);
        setSelectedPath(newPath);
        setSelectedConcretePath(newConcretePath);

        onValidData(newConcretePath, amountIsValid);
      }
    },
    [
      amountIsValid,
      collectionId,
      factRefs,
      factValidity,
      onValidData,
      pathPrefix,
      selectedConcretePath,
      selectedPath,
      setCodeOptionValue,
    ]
  );

  const i18nKey = [`fields.${parentPath}./amount`, `fields.generics.boxCode.amount`];
  const boxCodeControlId = useId();
  const boxAmountControlId = useId();

  const codeKeyLower = `fields.${selectedPath ?? `generics.boxCode.code`}.name_lower`;

  return (
    <Fieldset className={styles[`boxFact`]}>
      <FormGroup error={showFeedback && !codeOptionValue}>
        <FormFieldWrapper
          labelKey={[`fields.${parentPath}./code`, `fields.generics.boxCode.code.name`]}
          collectionId={collectionId}
          controlId={boxCodeControlId}
          errorMessage={t(`enums.messages.requiredField`)}
          labelContext={{ number: index + 1 }}
          showError={showFeedback && !codeOptionValue}
        >
          {/* TODO: Can we share logic with `Enum`? */}
          <Select
            defaultValue={codeOptionValue}
            id={boxCodeControlId}
            name={`${parentPath}.code`}
            onChange={onPathChange}
            inputRef={setInitialFocus}
            disabled={readOnly}
          >
            <option value='' disabled={!!codeOptionValue}>
              {t(`select.select`)}
            </option>
            {codeOptions
              .filter((code) => codeOptionValue === code || !selectedCodeOptions.includes(code))
              .map((option) => {
                const translationKey = `fields.${optionsPath}.${option}`;
                const maybeCustomTranslationKey = `fields.${parentPath}.${optionsPath}.${option}`;
                return (
                  <option key={option} value={option}>
                    <Translation i18nKey={[maybeCustomTranslationKey, translationKey]} collectionId={collectionId} />
                  </option>
                );
              })}
          </Select>
        </FormFieldWrapper>
      </FormGroup>
      {!isBoolean && (
        <DollarImplementation
          {...props}
          fact={
            // Prevent the fact from initializing with an invalid value when switching from a boolean value
            typeof fact === `boolean` ? undefined : fact
          }
          setFact={setFact}
          clearFact={clearFact}
          labelConfig={selectedConcretePath === undefined ? { type: `config`, i18nKey } : undefined}
          path={selectedPath ?? parentPath}
          concretePath={selectedConcretePath ?? concreteParentPath}
          collectionId={collectionId}
          showFeedback={showFeedback}
          isValid={amountIsValid}
          onValidData={innerOnValidData}
          factValidity={factValidity}
          controlIdOverride={boxAmountControlId}
          ref={codeAmountRef}
          readOnly={readOnly}
          required
        />
      )}
      <Button
        type='button'
        onClick={() => {
          clearFact();
          onRemove();
        }}
        className='margin-top-1'
        unstyled
        disabled={readOnly}
      >
        <Translation
          i18nKey={[`fields.${parentPath}.delete`]}
          collectionId={collectionId}
          context={{ code: t(codeKeyLower) }}
        />
      </Button>
    </Fieldset>
  );
}

type BoxFactData = { code?: string; reactKey: string };

/*
  This `FactSelect` component provides a single-screen UI for managing a set of dollar facts.

  The `path` property should identify an ad hoc Enum path whose option values name the facts to be managed.
  The paths of the managed facts must match the component `path` until the last level, for example:
    component with path `/some/path/myEnum` can manage dollar or boolean fact paths `/some/path/this`,
    `/some/path/that`,etc.

  When a new box input is created, it is initialized with a select dropdown containing the translated labels
  for the managed facts and an empty dollar input.

  Dollar facts with existing values will be immediately rendered with their current value populating the input.
  New dollar facts will be initialized with an empty input instead.  If the code is changed on an existing value,
  the fact referenced by the previous code will be cleared and the current amount will be set on the newly selected
  fact, if the new fact is a dollar value.

  If the selected fact is a boolean, the dollar input will be hidden instead.  The fact will be set to "true" if present
  and cleared when removed.
*/
const FactSelect = forwardRef<HTMLInputElement, FactProps>(
  ({ path, onValidData, concretePath, collectionId, ...props }, ref) => {
    const { factGraph } = useFactGraph();
    const { optionsPath, currentValues, pathPrefix, enumOptionValues } = getCurrentFactSelectValues(
      path,
      collectionId,
      factGraph
    );

    const { t } = useTranslation();

    const [nextTempId, setNextTempId] = useState(0);

    const [shouldAllowInitialFocus, setShouldAllowInitialFocus] = useState(false);

    const [boxCodes, setBoxCodes] = useState<BoxFactData[]>(
      (() => {
        const initialValues = currentValues
          .filter(({ currentResult: { complete } }) => complete)
          .map(({ optionValue: code }, index) => ({ code, reactKey: `existing:${index}` }));

        // Initialize with a single, empty box if no codes are currently selected
        return initialValues.length > 0 ? initialValues : [{ code: undefined, reactKey: `new:init` }];
      })()
    );

    const boxCodeSelectRefs = useRef(new Map<string, MutableRefObject<HTMLSelectElement>>());

    const handleBoxRef = useCallback((optionValue: string) => {
      if (!boxCodeSelectRefs.current.has(optionValue)) {
        boxCodeSelectRefs.current.set(optionValue, createRef() as MutableRefObject<HTMLSelectElement>);
      }

      return boxCodeSelectRefs.current.get(optionValue) as MutableRefObject<HTMLSelectElement>;
    }, []);

    const selectedCodes = useMemo(
      () => boxCodes.map(({ code }) => code).filter((code): code is string => code !== undefined),
      [boxCodes]
    );

    const addBoxCode = useCallback(() => {
      setSmoothScroll(true);
      setBoxCodes((previousBoxCodes) => [...previousBoxCodes, { code: undefined, reactKey: `new:${nextTempId}` }]);
      setNextTempId(nextTempId + 1);
      setShouldAllowInitialFocus(true);
    }, [nextTempId]);

    const removeCode = useCallback(
      (reactKeytoRemove: string) => {
        setSmoothScroll(true);
        // eslint-disable-next-line eqeqeq
        const currentIndex = boxCodes.findIndex(({ reactKey }) => reactKeytoRemove == reactKey);
        const nextBoxCode: BoxFactData | undefined = boxCodes[currentIndex + 1];

        const focusTarget = nextBoxCode
          ? boxCodeSelectRefs.current.get(nextBoxCode.reactKey)?.current
          : addCodeRef.current;
        focusTarget?.focus();
        focusTarget?.scrollIntoView({ block: `center` });

        setBoxCodes((previousCodes) => previousCodes.filter(({ reactKey }) => reactKey !== reactKeytoRemove));
      },
      [boxCodes]
    );

    const setBoxCode = useCallback((code: string, reactKey: string) => {
      setBoxCodes((previousCodes) =>
        previousCodes.map((codeItem) => (codeItem.reactKey === reactKey ? { code, reactKey } : codeItem))
      );
    }, []);

    const addCodeRef = useRef<HTMLButtonElement>();

    useEffect(() => {
      onValidData(
        concretePath,
        // Array.every returns true if empty, which is semantically correct here
        boxCodes.every(({ code }) => code !== undefined)
      );

      // Forward the ref to the current proxy
      let proxyRef: MutableRefObject<HTMLSelectElement | HTMLButtonElement | undefined>;

      const firstPendingBoxCode = boxCodes.find(({ code }) => code === undefined)?.reactKey;
      const firstActualizedBoxCode = boxCodes[0]?.code;
      if (firstPendingBoxCode) {
        proxyRef = boxCodeSelectRefs.current.get(firstPendingBoxCode) as MutableRefObject<HTMLSelectElement>;
      } else if (firstActualizedBoxCode) {
        proxyRef = boxCodeSelectRefs.current.get(firstActualizedBoxCode) as MutableRefObject<HTMLSelectElement>;
      } else proxyRef = addCodeRef;

      // We have to cast here because our component builder logic breaks if we don't type the ref as an HTMLInputElement
      if (typeof ref === `function`) ref(proxyRef.current as HTMLInputElement);
      else if (proxyRef?.current && ref) ref.current = proxyRef.current as HTMLInputElement;
    }, [boxCodes, concretePath, onValidData, ref]);

    return (
      <>
        {boxCodes.length === 0 ? (
          <p className='margin-top-4 text-base-light'>{t(`fields.${path}.blank`)}</p>
        ) : undefined}
        {boxCodes.map(({ code, reactKey }, index) => {
          return (
            <BoxFact
              key={reactKey}
              index={index}
              codeOptionValue={code}
              codeOptions={enumOptionValues}
              selectedCodeOptions={selectedCodes}
              parentPath={path}
              concreteParentPath={concretePath}
              optionsPath={optionsPath}
              pathPrefix={pathPrefix}
              setCodeOptionValue={(codeOptionValue) => {
                setBoxCode(codeOptionValue, reactKey);
              }}
              collectionId={collectionId}
              onValidData={onValidData}
              onRemove={() => removeCode(reactKey)}
              codeSelectRef={handleBoxRef(reactKey)}
              shouldAllowInitialFocus={shouldAllowInitialFocus}
              {...props}
            />
          );
        })}

        <button
          ref={addCodeRef as LegacyRef<HTMLButtonElement>}
          type='button'
          onClick={addBoxCode}
          className='usa-button usa-button--outline margin-top-4'
          disabled={props.readOnly}
        >
          <Translation i18nKey={[`fields.${path}.add`]} collectionId={collectionId} />
        </button>
      </>
    );
  }
);

FactSelect.displayName = `FactSelect`;

export default FactSelect;
