export interface Persister {}

export declare const Persister: {
  // we have to take in a string instead of actualon.
  // this allows scala to do the deserialization into the object
  // prototypes that it expects.
  create: (existingFactonString: String) => Persister;
};
