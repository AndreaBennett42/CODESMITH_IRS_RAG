export type PersistenceWrapperTypeName = `gov.irs.factgraph.persisters.${string}`;
export type PersistenceWrapper<
  TypeName extends PersistenceWrapperTypeName = PersistenceWrapperTypeName,
  Item extends object | string | boolean = object | string | boolean
> = {
  $type: TypeName;
  item: Item;
};

export function createPersistenceWrapper<
  TypeName extends PersistenceWrapperTypeName,
  Item extends object | string | boolean
>(typeName: TypeName, item: Item): PersistenceWrapper<TypeName, Item> {
  return {
    $type: typeName,
    item,
  };
}

export function createEnumWrapper(value: string, optionsPath: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.EnumWrapper`, {
    value: [value],
    enumOptionsPath: optionsPath,
  });
}

export function createMultEnumWrapper(values: string[], optionsPath: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.MultEnumWrapper`, {
    values,
    enumOptionsPath: optionsPath,
  });
}

export function createBooleanWrapper(value: boolean) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.BooleanWrapper`, value);
}

export function createDayWrapper(date: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.DayWrapper`, { date });
}

export function createStringWrapper(value: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.StringWrapper`, value);
}

export function createTinWrapper(value: { area: string; group: string; serial: string }) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.TinWrapper`, value);
}

export function createEinWrapper(prefix: string, serial: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.EinWrapper`, { prefix: prefix, serial: serial });
}

export function createDollarWrapper(value: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.DollarWrapper`, value);
}

export function createCollectionWrapper(items: string[]) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.CollectionWrapper`, { items });
}

export function createCollectionItemWrapper(itemId: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.CollectionItemWrapper`, {
    id: itemId,
  });
}

export function createPinWrapper(value: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.PinWrapper`, { pin: value });
}

export function createIpPinWrapper(value: string) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.IpPinWrapper`, { pin: value });
}

export function createAddressWrapper(
  streetAddress = `123 main`,
  city = `city`,
  postalCode = `32013`,
  stateOrProvence = `FL`
) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.AddressWrapper`, {
    streetAddress: streetAddress,
    city: city,
    postalCode: postalCode,
    stateOrProvence: stateOrProvence,
    country: ``,
  });
}

export function createPhoneWrapper(areaCode = `555`, officeCode = `555`, lineNumber = `5555`) {
  return createPersistenceWrapper(`gov.irs.factgraph.persisters.E164Wrapper`, {
    $type: `gov.irs.factgraph.types.UsPhoneNumber`,
    areaCode: areaCode,
    officeCode: officeCode,
    lineNumber: lineNumber,
  });
}
