/*
 * Content Batching Logic
 * All of the metadata for content batches is contained in this file, mostly in
 * the CONTENT_BATCHES variable. Developers will keep this updated (with input
 * from design) and this will drive the all screens page where design and
 * product can work with external stakeholders to review/approve content.
 */
interface ScreenStatus {
  isOpen: boolean;
  isLocked: boolean;
  isPublishable: boolean;
}

interface BatchHistoryItem {
  status: BatchStatus;
  completed: string; // Should be a date formatted: yyyy-mm-dd
}

export interface BatchDetails {
  type: BatchType;
  status: BatchStatus;
  started: string; // should be a date formatted: yyyy-mm-dd
  history?: BatchHistoryItem[];
}

export enum BatchStates {
  LOCKED = `Locked`,
  UNLOCKED = `Unlocked`,
  PUBLISHABLE = `Publishable`,
  UNPUBLISHABLE = `Unpublishable`,
}

// A content batch can be in one of the following workflow states.
export enum BatchStatus {
  WIP = `WIP`,
  DF_REVIEW = `DF Review`,
  CC_REVIEW = `CC Review`,
  TRANSLATION_CHECK = `Translation Check`,
  COMPLETE = `Complete`,
}

// A content batch is either an Admin or Content batch.
// Content batches are the default. Admin batches are so designers can
// put certain screens in review on an ad hoc basis if needed.
export enum BatchType {
  ADMIN = `Admin`,
  CONTENT = `Content`,
}

export type ContentBatches = Record<string, BatchDetails>;

// Our content batches are defined here.
// Developers should edit this as new batches are added to the application.
// As a batch moves through the workflow, this structure will need to be
// updated. See batches.test.ts for example batches with the history
// attribute filled out.

export const CONTENT_BATCHES = {
  'savers-8880-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'schedule-b-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'schedule-b-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'alaska-permanent-fund-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'cdcc-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'dependent-care-benefits-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'cdcc-credits-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'retirement-1099R-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'retirement-1099R-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'retirement-1099R-2': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'retirement-1099R-3': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'retirement-1099R-4': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'edc-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'hsa-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'hsa-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'ptc-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'ptc-2': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'ptc-3': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'data-import-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'data-import-w2': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'income-assorted-kos-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'eitc-ids-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'cdcc-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'updates-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
    ],
  },
  'updates-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'information-architecture-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
  'cdcc-2': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'cdcc-3': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: ``,
      },
      {
        status: BatchStatus.WIP,
        completed: ``,
      },
    ],
  },
  'credits-ia-1': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: ``,
  },
} as const satisfies ContentBatches;

// Export list of our current content batches
export type BATCH_NAME = keyof typeof CONTENT_BATCHES;
export const BATCH_NAMES = Object.keys(CONTENT_BATCHES) as BATCH_NAME[];

// Look through a screens batches and return true if ANY of the batches are in the given status
export const screenIsInStatus = (status: BatchStates, batches: BATCH_NAME[]): boolean => {
  if (!batches || !status) {
    return false;
  }
  let isInState = false;
  const screenStatus = calculateScreenStatus(batches);
  // eslint-disable-next-line eqeqeq
  if (status == BatchStates.LOCKED && screenStatus.isLocked) {
    isInState = true;
    // eslint-disable-next-line eqeqeq
  } else if (status == BatchStates.UNLOCKED && !screenStatus.isLocked) {
    isInState = true;
    // eslint-disable-next-line eqeqeq
  } else if (status == BatchStates.PUBLISHABLE && screenStatus.isPublishable) {
    isInState = true;
    // eslint-disable-next-line eqeqeq
  } else if (status == BatchStates.UNPUBLISHABLE && !screenStatus.isPublishable) {
    isInState = true;
  } else {
    isInState = false;
  }
  return isInState;
};

// Look through a screens batches and return true if ANY of the batches are in the given workflow step
export const screenIsInWorkflowStep = (step: BatchStatus, batches: BATCH_NAME[]): boolean => {
  if (!batches || !step) {
    return false;
  }
  let isInWorkflow = false;
  for (let i = 0; i < batches.length; i++) {
    // eslint-disable-next-line eqeqeq
    if (CONTENT_BATCHES[batches[i]].status == step) {
      isInWorkflow = true;
    }
  }
  return isInWorkflow;
};

// Calculates the status for a given screen, which could be part of
// multiple content batches.
export const calculateScreenStatus = (
  batchNames: string[] = [],
  contentBatches: ContentBatches = CONTENT_BATCHES
): ScreenStatus => {
  // defaults
  let isOpen = false;
  let isLocked = false;
  let isPublishable = true;

  for (let i = 0; i < batchNames.length; i++) {
    const batchName = batchNames[i];
    const batch = contentBatches[batchName];

    // Default status for a screen with no batches
    if (!batch) {
      return { isOpen, isLocked, isPublishable };
    }

    // if any statuses are anything other than complete, the screen
    // is not publishable and is open
    if (batch.status !== BatchStatus.COMPLETE) {
      isOpen = true;
      isPublishable = false;
    }

    // if any statuses are in any of the review states, the screen is locked
    if (
      // eslint-disable-next-line eqeqeq
      batch.status == BatchStatus.DF_REVIEW ||
      // eslint-disable-next-line eqeqeq
      batch.status == BatchStatus.CC_REVIEW ||
      // eslint-disable-next-line eqeqeq
      batch.status == BatchStatus.TRANSLATION_CHECK
    ) {
      isLocked = true;
    }
  }

  return { isOpen, isLocked, isPublishable };
};
