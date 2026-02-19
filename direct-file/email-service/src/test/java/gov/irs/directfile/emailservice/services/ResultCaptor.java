package gov.irs.directfile.emailservice.services;

import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

// ResultCaptor to make it easier to capture return values in tests.  See:
//   - https://stackoverflow.com/questions/7095871
//   - https://stackoverflow.com/questions/7254200
public class ResultCaptor<T> implements Answer {
    private T result = null;

    public T getResult() {
        return result;
    }

    @Override
    public T answer(InvocationOnMock invocationOnMock) throws Throwable {
        result = (T) invocationOnMock.callRealMethod();
        return result;
    }
}
