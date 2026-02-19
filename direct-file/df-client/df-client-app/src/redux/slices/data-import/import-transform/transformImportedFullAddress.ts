import { AddressFactory } from '@irs/js-factgraph-scala';
import { AboutYouBasicAddress } from '../schema/AboutYouBasicSchema.js';
import { W2Address } from '../schema/W2Schema.js';

const BLANK_ADDRESS = {
  mailingAddress: ``,
  streetAddress: ``,
  city: ``,
  postalCode: ``,
  stateOrProvence: ``,
  streetAddressLine2: ``,
};

export function transformImportedFullAddress(input: AboutYouBasicAddress): AboutYouBasicAddress {
  const { streetAddress, city, postalCode, stateOrProvence, streetAddressLine2 } = input;
  const streetAddressLine1 = input.mailingAddress ? input.mailingAddress + ` ` + streetAddress : streetAddress;
  const result = AddressFactory(streetAddressLine1, city, postalCode, stateOrProvence, streetAddressLine2 || ``, `USA`);

  return result.right ? input : BLANK_ADDRESS;
}

const BLANK_W2_ADDRESS = {
  streetAddressLine1: ``,
  city: ``,
  zipCode: ``,
  state: ``,
  streetAddressLine2: ``,
};

export function transformImportedFullW2Address(input: W2Address): W2Address {
  const { streetAddressLine1, city, zipCode, state, streetAddressLine2 } = input;
  const result = AddressFactory(streetAddressLine1, city, zipCode, state, streetAddressLine2 || ``, `USA`);

  return result.right ? input : BLANK_W2_ADDRESS;
}

export const isValidAddress = (address: W2Address): boolean => {
  if (!address.streetAddressLine1) {
    return false;
  }
  if (!address.city) {
    return false;
  }
  if (!address.zipCode) {
    return false;
  }
  if (!address.state) {
    return false;
  }
  return true;
};
