package gov.irs.directfile.submit;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import gov.irs.directfile.submit.domain.ActionQueue;
import gov.irs.directfile.submit.service.ActionHandler;
import gov.irs.directfile.submit.service.OfflineModeService;

@Slf4j
@Service
public class Runner {

    // This is the message queue of actions
    // It could be added to by background threads (from timers)
    // but should only be read in one place.
    ActionQueue actions;

    private final ActionHandler actionHandler;
    private final OfflineModeService offlineModeService;

    public Runner(ActionQueue actions, ActionHandler actionHandler, OfflineModeService offlineModeService) {
        this.actions = actions;
        this.actionHandler = actionHandler;
        this.offlineModeService = offlineModeService;
    }

    @PostConstruct
    public void setup() {
        this.Start();
    }

    void Start() {
        Runnable runnable = this::handleActions;
        log.info("starting message loop");
        new Thread(runnable).start();
    }

    private void handleActions() {
        try {
            while (true) {
                step();
            }
        } catch (InterruptedException e) {
            log.error("Exiting", e);
        }
    }

    public void step() throws InterruptedException {
        if (!offlineModeService.isOfflineModeEnabled()) {
            if (actions.getInProgressActions().isEmpty()) {
                log.info("Handling new action");
                var action = actions.getNewActions().take();
                actionHandler.handleAction(action);
            } else {
                log.info("Handling in-progress action");
                var action = actions.getInProgressActions().take();
                actionHandler.handleAction(action);
            }
        }
    }
}
