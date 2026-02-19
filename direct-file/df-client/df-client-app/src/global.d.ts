/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable no-var */
export {};

type EgainDockChat = {
  openHelp: () => void;
};

declare global {
  var debugFactGraph: any;
  var debugFacts: any;
  var debugFactGraphMeta: any;
  var debugScalaFactGraphLib: any;
  var loadFactGraph: (json: string) => void;
  var saveFactGraphToLocalStorageKey: (keyId: string, force?: boolean) => void;
  var loadFactGraphFromLocalStorageKey: (keyId: string) => void;
  const egainDockChat: EgainDockChat;
}
