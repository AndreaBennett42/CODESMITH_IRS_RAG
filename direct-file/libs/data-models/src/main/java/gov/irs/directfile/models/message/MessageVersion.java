package gov.irs.directfile.models.message;

/**
 * The sole purpose of placing this version class in a shared library is to
 * establish a version naming contract between producers and consumers. It does
 * not imply that any specific version is supported. Each service independently
 * determines which versions it supports.
 *
 * This list can be expanded as needed.
 *
 * This class defines message version constants as static final strings instead
 * of using an enum. The reason for this approach is to ensure that these
 * constants can be used in contexts where only compile-time constants are
 * allowed, such as annotation attributes, e.g.
 * VersionedMessageHandlerQualifier.
 *
 * Unlike enum constants, which require method calls (e.g., name()) to retrieve
 * their values, these static final strings are directly usable as compile-time
 * constants, providing greater flexibility in annotation-based configurations.
 */
public class MessageVersion {
    public static final String VERSION_1 = "1.0";
    public static final String VERSION_2 = "2.0";
    public static final String VERSION_3 = "3.0";
    public static final String VERSION_4 = "4.0";
    public static final String VERSION_5 = "5.0";
    public static final String VERSION_6 = "6.0";
    public static final String VERSION_7 = "7.0";
    public static final String VERSION_8 = "8.0";
    public static final String VERSION_9 = "9.0";
}
