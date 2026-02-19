import { BatchStatus, BatchType, calculateScreenStatus, ContentBatches } from './batches.js';

// Mock Content Batches for testing purposes only! These
// statuses aren't real and set only to exercise unit tests.
const MOCK_BATCHES = {
  'savers-8880-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.CC_REVIEW,
    started: `2024-09-15`,
    history: [
      {
        status: BatchStatus.DF_REVIEW,
        completed: `2024-09-15`,
      },
      {
        status: BatchStatus.WIP,
        completed: `2024-09-13`,
      },
    ],
  },
  'schedule-b-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.WIP,
    started: `2024-10-02`,
  },
  'alaska-permanent-fund-0': {
    type: BatchType.CONTENT,
    status: BatchStatus.COMPLETE,
    started: `2024-10-02`,
    history: [
      {
        status: BatchStatus.TRANSLATION_CHECK,
        completed: `2024-09-18`,
      },
      {
        status: BatchStatus.CC_REVIEW,
        completed: `2024-09-17`,
      },
      {
        status: BatchStatus.DF_REVIEW,
        completed: `2024-09-15`,
      },
      {
        status: BatchStatus.WIP,
        completed: `2024-09-13`,
      },
    ],
  },
} satisfies ContentBatches;

describe(`Batches`, async () => {
  it(`Calculates screen status correctly`, () => {
    // test WIP
    let status = calculateScreenStatus([`schedule-b-0`], MOCK_BATCHES);
    expect(status.isOpen).toBe(true);
    expect(status.isPublishable).toBe(false);
    expect(status.isLocked).toBe(false);

    // test COMPLETE
    status = calculateScreenStatus([`alaska-permanent-fund-0`], MOCK_BATCHES);
    expect(status.isOpen).toBe(false);
    expect(status.isPublishable).toBe(true);
    expect(status.isLocked).toBe(false);

    // test CC_REVIEW
    status = calculateScreenStatus([`savers-8880-0`], MOCK_BATCHES);
    expect(status.isOpen).toBe(true);
    expect(status.isPublishable).toBe(false);
    expect(status.isLocked).toBe(true);

    // test having a WIP + CC_REVIEW
    status = calculateScreenStatus([`savers-8880-0`, `schedule-b-0`], MOCK_BATCHES);
    expect(status.isOpen).toBe(true);
    expect(status.isPublishable).toBe(false);
    expect(status.isLocked).toBe(true);

    // test having a WIP + COMPLETE
    status = calculateScreenStatus([`schedule-b-0`, `alaska-permanent-fund-0`], MOCK_BATCHES);
    expect(status.isOpen).toBe(true);
    expect(status.isPublishable).toBe(false);
    expect(status.isLocked).toBe(false);

    // test having an empty query
    status = calculateScreenStatus([], MOCK_BATCHES);
    expect(status.isOpen).toBe(false);
    expect(status.isPublishable).toBe(true);
    expect(status.isLocked).toBe(false);

    // test having an empty query
    const b = undefined;
    status = calculateScreenStatus(b, MOCK_BATCHES);
    expect(status.isOpen).toBe(false);
    expect(status.isPublishable).toBe(true);
    expect(status.isLocked).toBe(false);
  });
});
