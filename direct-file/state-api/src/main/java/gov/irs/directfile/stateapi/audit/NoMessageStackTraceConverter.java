package gov.irs.directfile.stateapi.audit;

import ch.qos.logback.classic.pattern.ThrowableProxyConverter;
import ch.qos.logback.classic.spi.IThrowableProxy;
import ch.qos.logback.classic.spi.ThrowableProxyUtil;
import ch.qos.logback.core.CoreConstants;

/*
 * https://logback.qos.ch/apidocs/src-html/ch/qos/logback/classic/pattern/RootCauseFirstThrowableProxyConverter.html
 *
 * This is a slightly modified version of the linked class
 * that omits the exception message from the stack trace.
 */

@SuppressWarnings("PMD.AvoidReassigningParameters")
public class NoMessageStackTraceConverter extends ThrowableProxyConverter {
    @Override
    protected String throwableProxyToString(IThrowableProxy tp) {
        StringBuilder buf = new StringBuilder(BUILDER_CAPACITY);
        recursiveAppendRootCauseFirst(buf, null, ThrowableProxyUtil.REGULAR_EXCEPTION_INDENT, tp);
        return buf.toString();
    }

    protected void recursiveAppendRootCauseFirst(StringBuilder sb, String prefix, int indent, IThrowableProxy tp) {
        if (tp.getCause() != null) {
            recursiveAppendRootCauseFirst(sb, prefix, indent, tp.getCause());
            prefix = null;
        }
        ThrowableProxyUtil.indent(sb, indent - 1);
        if (prefix != null) {
            sb.append(prefix);
        }

        sb.append(tp.getClassName());
        sb.append(CoreConstants.LINE_SEPARATOR);
        subjoinSTEPArray(sb, indent, tp);
        IThrowableProxy[] suppressed = tp.getSuppressed();
        if (suppressed != null) {
            for (IThrowableProxy current : suppressed) {
                recursiveAppendRootCauseFirst(
                        sb, CoreConstants.SUPPRESSED, indent + ThrowableProxyUtil.SUPPRESSED_EXCEPTION_INDENT, current);
            }
        }
    }
}
