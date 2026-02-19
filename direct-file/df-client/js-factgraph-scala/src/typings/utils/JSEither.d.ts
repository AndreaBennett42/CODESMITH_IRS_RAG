export interface EitherL<L> {
  isLeft: true;
  left: L;
}

export interface EitherR<R> {
  isRight: true;
  right: R;
}

export interface JSeither<L, R> extends EitherR<R>, EitherL<L> {
  mapLeftRight: (onInvalid: (phoneNumber: L) => void, onValid: (phoneNumber: R) => void) => void;
}
