export {};

type EgainDockChat = {
  EntryPointId: string;
  openHelp: () => void;
};

/*
  Defining the global variable that become available after loading the following eGain scripts:
  custoffers.js and launchDirectFile.js. These are loaded via script import in our HTML.
*/
declare global {
  const egainDockChat: EgainDockChat;
}
