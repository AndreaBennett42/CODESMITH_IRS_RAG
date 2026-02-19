import { FactGraphResult, formatAddressForHTML } from '@irs/js-factgraph-scala';
import { Path } from '../fact-dictionary/Path.js';
import { FactConfig } from '../flow/ContentDeclarations.js';

export const DOLLAR_FORMAT_PARAMS: Intl.NumberFormatOptions = {
  style: `currency`,
  currency: `USD`,
};
export const DATE_FORMAT_PARAMS: Intl.DateTimeFormatOptions = {
  dateStyle: `long`,
};

export function prettyPrintFactGraphValues(
  rawPath: Path,
  // It would be nice if fact graph types were better
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: FactGraphResult<any>,
  t: (path: string | string[]) => string,
  enumOptioni18nPrefix: string,
  factConfig?: FactConfig
) {
  const formatParams: Partial<Record<Path, unknown>> = {};
  let output: string | number | Date = ``;
  if (data && data.hasValue) {
    switch (data.typeName) {
      case `class scala.math.BigDecimal`:
        // All `BigDecimal`s are currently dollars
        output = Number(data.get.toString());
        formatParams[rawPath] = DOLLAR_FORMAT_PARAMS;
        if (factConfig?.componentName === `Dollar` && factConfig.props.hasZeroOverride && output === 0) {
          output = t([`${enumOptioni18nPrefix}.${rawPath}.zeroValue`, `${enumOptioni18nPrefix}.zeroValue`]);
        }
        break;
      case `class gov.irs.factgraph.types.Enum`: {
        const enumValue = data.get.getValue() as string;
        const optionsPath = data.get.getEnumOptionsPath();
        output = t([
          `${enumOptioni18nPrefix}.${rawPath}.${optionsPath}.${enumValue}`,
          `${enumOptioni18nPrefix}.${optionsPath}.${enumValue}`,
        ]);
        break;
      }
      case `class gov.irs.factgraph.types.MultiEnum`: {
        const enumValues = data.get.getValue() as Set<string>;
        const optionsPath = data.get.getEnumOptionsPath();

        output = Object.values(enumValues)
          .map((value) =>
            t([
              `${enumOptioni18nPrefix}.${rawPath}.${optionsPath}.${value}`,
              `${enumOptioni18nPrefix}.${optionsPath}.${value}`,
            ])
          )
          .join(`, `);
        break;
      }
      case `class gov.irs.factgraph.types.Address`: {
        output = formatAddressForHTML(data.get);
        break;
      }
      case `class gov.irs.factgraph.types.Day`: {
        const { day, month, year } = data.get;
        output = new Date(year, month - 1, day);
        formatParams[rawPath] = DATE_FORMAT_PARAMS;
        break;
      }
      case `class gov.irs.factgraph.types.UsPhoneNumber`: {
        const cleaned = (`` + data.get.toString()).replace(/\D/g, ``);
        const match = cleaned.match(/^(1|)?(\d{3})(\d{3})(\d{4})$/);
        if (match) {
          output = `(` + match[2] + `) ` + match[3] + `-` + match[4];
        }
        break;
      }
      // We may want to replace this case with a change in the scala code so that we can better identify this scenario
      // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/3207 was created to track this
      case `class java.lang.String`: {
        output = data.get.toString();
        if (rawPath === `/bankAccount/accountType`) {
          output = t(`${enumOptioni18nPrefix}.${rawPath}.${output}`);
        }
        break;
      }
      default:
        output = data.get.toString();
    }
  }
  return { output, formatParams };
}
