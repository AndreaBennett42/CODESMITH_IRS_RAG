package gov.irs.directfile.submit.service;

import java.util.concurrent.atomic.AtomicBoolean;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class OfflineModeService {

    @Value("${submit.should-disable-mef-connectivity}")
    @Getter
    private Boolean shouldStayOffline = false;

    private final AtomicBoolean offlineModeEnabled = new AtomicBoolean(false);

    public OfflineModeService() {}

    public boolean isOfflineModeEnabled() {
        return offlineModeEnabled.get();
    }

    public void disableOfflineMode() {
        this.offlineModeEnabled.compareAndSet(true, false);
    }

    public void enableOfflineMode() {
        if (!isOfflineModeEnabled()) {
            this.offlineModeEnabled.compareAndSet(false, true);
        }
    }
}
