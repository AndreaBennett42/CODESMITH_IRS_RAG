import { ConcretePath, Fact, FactGraph, SaveReturnValue, ScalaList, PersisterSyncIssue } from '@irs/js-factgraph-scala';

class FakeResult<T> {
  private _value: T;
  constructor(value: T) {
    this._value = value;
  }
  get get(): T {
    if (this._value === null) {
      throw new Error(`attempted to retrieve the value of an incomplete result`);
    }
    if (this._value instanceof FakePlaceholder) {
      return this._value.get();
    }
    return this._value;
  }
  get complete(): boolean {
    return this._value !== null && this._value !== undefined && !(this._value instanceof FakePlaceholder);
  }

  get toString(): string {
    const status = this.complete ? `complete` : `incomplete`;
    return `Result(${this._value}, ${status})`;
  }

  get hasValue(): boolean {
    if (this._value instanceof FakePlaceholder) {
      return this._value._hasValue();
    } else {
      return this._value !== null && this._value !== undefined;
    }
  }

  get typeName(): string {
    throw new Error(`not implemented`);
  }
}

export class FakePlaceholder<T> {
  private _value: T;
  constructor(value: T) {
    this._value = value;
  }
  get() {
    return this._value;
  }

  _hasValue() {
    return this._value !== null && this._value !== undefined;
  }
}

export class FakeFactGraph implements FactGraph {
  private _values: Map<string, boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>;
  constructor(values: Map<string, boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>) {
    this._values = values;
  }

  set(_key: ConcretePath) {
    const value = this._values.set(_key, null);
    if (value !== undefined) {
      return new FakeResult<boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>(value);
    } else {
      return new FakeResult<boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>(null);
    }
  }

  delete(_key: string) {
    throw new Error(`not implemented`);
  }

  getDictionary = () => {
    throw new Error(`Not implemented`);
  };

  save(): SaveReturnValue {
    return { valid: true, limitViolations: [] };
  }

  getEnumOptions = (_enumId: string) => {
    throw new Error(`Not implemented`);
  };

  toJSON(): string {
    throw new Error(`Not Implemented`);
  }

  toJson(): string {
    throw new Error(`Not Implemented`);
  }

  toStringDictionary() {
    throw new Error(`Not implemented`);
  }

  explainAndSolve(_path: ConcretePath): string[][] {
    throw new Error(`Not implemented`);
  }

  get(path: string): FakeResult<boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>> {
    const value = this._values.get(path);
    if (value !== undefined) {
      return new FakeResult<boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>(value);
    } else {
      return new FakeResult<boolean | null | FakePlaceholder<boolean | null> | ScalaList<string>>(null);
    }
  }
  getFact(_path: ConcretePath): Fact {
    throw new Error(`Not implemented`);
  }

  checkPersister(): PersisterSyncIssue[] {
    throw new Error(`Not implemented`);
  }
}
