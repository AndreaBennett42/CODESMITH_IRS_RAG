package gov.irs.directfile.submit.service.startup.runners;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import gov.irs.directfile.submit.service.polling.MeFHealthPoller;

import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
class MefConnectivityApplicationStartRunnerTest {
    MefConnectivityApplicationStartRunner mefConnectivityApplicationStartRunner;

    @Mock
    MeFHealthPoller meFHealthPoller;

    @BeforeEach
    public void setup() {
        mefConnectivityApplicationStartRunner = new MefConnectivityApplicationStartRunner(meFHealthPoller);
    }

    @Test
    public void run_shouldCall_meFHealthPollerPerformMefConnectivityCheckMethod() throws Exception {
        mefConnectivityApplicationStartRunner.run(null);

        verify(meFHealthPoller, times(1)).performMefConnectivityCheck();
    }
}
