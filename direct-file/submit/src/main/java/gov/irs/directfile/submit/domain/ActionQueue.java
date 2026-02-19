package gov.irs.directfile.submit.domain;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;
import lombok.Getter;
import lombok.Setter;

import gov.irs.directfile.submit.command.Action;

@Getter
@Setter
@SuppressFBWarnings(
        value = {"EI_EXPOSE_REP"},
        justification = "Initial SpotBugs Setup")
public class ActionQueue {
    private BlockingQueue<Action> inProgressActions = new LinkedBlockingQueue<>();
    private BlockingQueue<Action> newActions = new LinkedBlockingQueue<>();
}
