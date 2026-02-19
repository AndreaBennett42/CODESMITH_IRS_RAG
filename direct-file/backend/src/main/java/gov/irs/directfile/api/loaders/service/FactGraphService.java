package gov.irs.directfile.api.loaders.service;

import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import scala.collection.Seq;
import scala.jdk.CollectionConverters;

import gov.irs.factgraph.FactDictionary;
import gov.irs.factgraph.Graph;
import gov.irs.factgraph.PersisterSyncIssue;
import gov.irs.factgraph.limits.LimitViolation;
import gov.irs.factgraph.monads.Result;
import gov.irs.factgraph.persisters.InMemoryPersister;
import gov.irs.factgraph.persisters.InMemoryPersisterJava;

import gov.irs.directfile.api.errors.FactGraphParseException;
import gov.irs.directfile.api.loaders.domain.*;
import gov.irs.directfile.api.loaders.errors.FactGraphSaveException;
import gov.irs.directfile.api.loaders.processor.FactGraphLoader;
import gov.irs.directfile.api.loaders.processor.XmlProcessor;
import gov.irs.directfile.models.EvaluatedFactInfo;
import gov.irs.directfile.models.FactEvaluationResult;
import gov.irs.directfile.models.FactTypeWithItem;

@Service
@Slf4j
public class FactGraphService {
    public static final String ABSTRACT_PATH_UUID_PATTERN = "#\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}";
    public static final Pattern ABSTRACT_PATH_UUID_REGEX_PATTERN = Pattern.compile(ABSTRACT_PATH_UUID_PATTERN);

    private static final String JAVA_STRING = "java.lang.String";
    private static final String JAVA_BOOLEAN = "java.lang.Boolean";
    private static TaxDictionaryDigest _digest;
    private static Set<String> exportZeroDollarFacts;
    private static FactDictionary _factDictionary;

    private final ApplicationContext applicationContext;
    private final ObjectMapper _objectMapper;

    @Value("${direct-file.loader.load-at-startup}")
    private Boolean loadAtStartup;

    @Value("${direct-file.loader.fact-dictionary-xml-pattern}")
    private String factDictionaryXmlPattern;

    public FactGraphService(final ObjectMapper objectMapper, final ApplicationContext applicationContext) {
        _objectMapper = objectMapper;
        this.applicationContext = applicationContext;
    }

    @PostConstruct
    @SuppressWarnings("PMD.UnusedPrivateMethod") // prevent init() being warned for 'UnusedPrivateMethod'
    private void init() throws IOException {
        if (!loadAtStartup) {
            log.warn("Loading fact graph configuration at startup disabled.  To enable loading, set"
                    + " the config variable \"direct-file.loader.load-at-startup\" to true.");
            _digest = new TaxDictionaryDigest("fact graph loading disabled", Map.of());
            _factDictionary = new FactDictionary();
            return;
        }
        log.info("Loading fact graph configuration from resource directory \"{}\"", factDictionaryXmlPattern);
        Resource[] resources = applicationContext.getResources(factDictionaryXmlPattern);
        _digest = new XmlProcessor().process(factDictionaryXmlPattern, resources);
        exportZeroDollarFacts = _digest.getExportZeroFacts();

        _factDictionary = new FactGraphLoader().createFactDictionary(_digest);
        log.info("Fact graph initialization complete");
    }

    public TaxDictionaryDigest getDigest() {
        return _digest;
    }

    public Graph getGraph(final Map<String, FactTypeWithItem> inputPersisterStateMap) {
        try {
            final String inputPersisterStateString = _objectMapper.writeValueAsString(inputPersisterStateMap);
            return new Graph(_factDictionary, InMemoryPersister.apply(inputPersisterStateString));
        } catch (Exception e) {
            throw new FactGraphParseException(e);
        }
    }

    public boolean factsParseCorrectly(final Map<String, FactTypeWithItem> inputPersisterStateMap) {
        boolean isCorrect = true;

        try {
            getGraph(inputPersisterStateMap);
        } catch (Exception e) {
            isCorrect = false;
        }

        return isCorrect;
    }

    public boolean hasSubmissionBlockingFacts(final Graph factGraph) {
        // If any fact marked as "<BlockSubmissionOnTrue />" in the digest is true, this
        // method will return true

        for (TaxFact fact : _digest.getSubmissionBlockingFacts()) {
            var result = factGraph.get(fact.path());
            if (result.hasValue() && result.get().equals(true)) {
                return true;
            }
        }
        return false;
    }

    public GraphSetResult setFacts(final Map<String, Object> writableValues) {
        InMemoryPersister persister = InMemoryPersisterJava.create();
        Graph factGraph = new Graph(_factDictionary, persister);

        List<LimitViolationInfo> limitViolationInfos = new ArrayList<>();

        writableValues.forEach((k, v) -> {
            factGraph.set(k, v);
            var saveResult = factGraph.save();

            boolean saveSuccessful = (boolean) saveResult._1();
            if (!saveSuccessful) {
                LimitViolation limitViolation = CollectionConverters.IterableHasAsJava(saveResult._2())
                        .asJava()
                        .iterator()
                        .next();
                limitViolationInfos.add(new LimitViolationInfo(k, v, limitViolation));
            }
        });

        return new GraphSetResult(factGraph, limitViolationInfos);
    }

    public Map<String, GraphGetResult> getFacts(final Graph factGraph, final Iterable<String> requestedOutputPaths) {
        Map<String, GraphGetResult> results = new HashMap<>();
        for (String outputPath : requestedOutputPaths) {
            try {
                var value = factGraph.get(outputPath);
                results.put(outputPath, new GraphGetResult(outputPath, value.get(), null));
            } catch (UnsupportedOperationException e) {
                results.put(outputPath, new GraphGetResult(outputPath, null, e));
            }
        }
        return results;
    }

    public GraphGetResult getFact(
            final Map<String, FactTypeWithItem> inputPersisterStateMap, final String requestedOutputPath) {
        try {
            final String inputPersisterStateString = _objectMapper.writeValueAsString(inputPersisterStateMap);
            Graph factGraph = new Graph(_factDictionary, InMemoryPersister.apply(inputPersisterStateString));
            var value = factGraph.get(requestedOutputPath);
            return new GraphGetResult(requestedOutputPath, value.get(), null);
        } catch (Exception e) {
            return new GraphGetResult(requestedOutputPath, null, e);
        }
    }

    public FactEvaluationResult extractFacts(final Set<String> factPaths, final Graph graph)
            throws JsonProcessingException, FactGraphSaveException {
        return extractFacts(factPaths, graph, false);
    }

    public FactEvaluationResult extractFacts(final Set<String> factPaths, final Graph graph, final boolean forXml)
            throws JsonProcessingException, FactGraphSaveException {
        return extractFacts(factPaths, graph, forXml, false, false);
    }

    public FactEvaluationResult extractFacts(
            final Set<String> factPaths,
            final Graph graph,
            final boolean forXml,
            final boolean extractIncompleteFacts,
            final boolean extractAllZeroDollarValues)
            throws JsonProcessingException, FactGraphSaveException {
        final var facts = new FactEvaluationResult();

        if (factPaths == null || factPaths.isEmpty()) {
            return facts;
        }

        // check consistency of previously persisted facts with the fact dictionary
        var persistedDataIssues = CollectionConverters.IterableHasAsJava(graph.checkPersister())
                .asJava()
                .iterator();

        boolean hasPersistedDataIssues = false;

        while (persistedDataIssues.hasNext()) {
            hasPersistedDataIssues = true;
            // Problems here do not prevent the submission to e-filing.
            PersisterSyncIssue issue = persistedDataIssues.next();
            log.warn("Persisted data issue at fact path " + issue.path() + ": " + issue.message());
        }

        // check for limit violations by saving graph
        var saveResult = graph.save();
        boolean saveSuccessful = (boolean) saveResult._1();
        if (!saveSuccessful) {
            var limitViolations = CollectionConverters.IterableHasAsJava(saveResult._2)
                    .asJava()
                    .iterator();
            while (limitViolations.hasNext()) {
                // Problems here prevent the submission to e-filing.
                LimitViolation v = limitViolations.next();
                log.warn("Limit violation at fact path " + v.factPath() + ": " + v.limitName());
            }

            throw new FactGraphSaveException(
                    hasPersistedDataIssues
                            ? "Has persisted data issues and limit violations."
                            : "Has limit violations.");
        }

        factPaths.forEach((String factPath) -> {
            final int indexOfEndOfCollectionName = factPath.indexOf("/*");
            if (indexOfEndOfCollectionName >= 0) {
                // Evaluate this wildcard path for each item in the collection ...
                try {
                    final Seq<String> collectionPaths = graph.getCollectionPaths(factPath);
                    Iterator<String> pathIter = CollectionConverters.IterableHasAsJava(collectionPaths)
                            .asJava()
                            .iterator();
                    while (pathIter.hasNext()) {
                        evaluateAndStoreFact(
                                graph,
                                pathIter.next(),
                                facts,
                                forXml,
                                extractIncompleteFacts,
                                extractAllZeroDollarValues);
                    }
                } catch (UnsupportedOperationException e) {
                    // This happens in test when no fact dictionary is loaded.
                    log.warn("Unable to get graph's collection paths: " + e.getMessage());
                }
                // ... and also evaluate the collection itself.
                factPath = factPath.substring(0, indexOfEndOfCollectionName);
            }

            evaluateAndStoreFact(graph, factPath, facts, forXml, extractIncompleteFacts, extractAllZeroDollarValues);
        });
        return facts;
    }

    private void evaluateAndStoreFact(
            final Graph graph,
            final String factPath,
            final FactEvaluationResult facts,
            final boolean forXml,
            final boolean extractIncompleteFacts,
            final boolean extractAllZeroDollarValues) {
        if (facts.getOptional(factPath).isPresent()) return;

        // Strip this from type names
        final String classPrefix = "class ";
        try {
            final Result<Object> result = graph.get(factPath);
            if (!result.complete()) {
                if (extractIncompleteFacts) {
                    // incomplete facts are extracted with a `null` value
                    // Is using `null` as the type questionable? Yes.
                    // I'd argue there isn't a better option, however.
                    facts.put(factPath, new EvaluatedFactInfo(null, null));
                }
                return;
            }
            final Object value = result.get();
            String typeName = result.typeName().trim();
            if (typeName.startsWith(classPrefix)) typeName = typeName.substring(classPrefix.length());
            if ("gov.irs.factgraph.types.Collection".equals(typeName)) {
                // This was added to maintain the collection ordering all the way through to
                // MeF.
                // There is an issue with serializing the scala iterable with jackson.
                // It is a rather involved deep fix, and this only currently applies
                // to collections. If this ever comes up for another type we will need
                // to modify the vector type in the fact graph to explain to the
                // serializer how to operate on it.

                // Convert the scala iterable to a Java List of UUIDs, and store that.
                final Seq<UUID> uuidSeq =
                        ((gov.irs.factgraph.types.Collection) value).items().toSeq();
                final List<UUID> uuidList =
                        CollectionConverters.SeqHasAsJava(uuidSeq).asJava();
                facts.put(factPath, new EvaluatedFactInfo(typeName, uuidList));
                return;
            } else if ("scala.math.BigDecimal".equals(typeName)
                    && value != null
                    && !extractAllZeroDollarValues
                    && suppressZeroDollarValue(factPath)) {
                try {
                    Double doubleValue = Double.parseDouble(value.toString());
                    if (doubleValue.equals(Double.valueOf(0))) return;
                } catch (NumberFormatException ex) {
                    // this is a really strange problem... but we know it isn't 0.
                    log.error("tried to parse a scala big decimal that wasn't a decimal somehow", ex);
                }
            }
            facts.put(factPath, factInfoFactory(typeName, value, forXml));
        } catch (UnsupportedOperationException e) {
            log.warn("Unable to evaluate fact with nonexistent path " + factPath);
        }
    }

    private EvaluatedFactInfo factInfoFactory(final String typeName, final Object factValue, final boolean forXml) {
        if (!forXml) return new EvaluatedFactInfo(typeName, factValue);

        final Object factClass = factValue.getClass();

        if (factClass == scala.math.BigDecimal.class) {
            return new EvaluatedFactInfo(JAVA_STRING, factValue.toString().split(Pattern.quote("."))[0]);
        }
        if (factClass == gov.irs.factgraph.types.UsPhoneNumber.class) {
            var phone = (gov.irs.factgraph.types.UsPhoneNumber) factValue;
            return new EvaluatedFactInfo(JAVA_STRING, phone.areaCode() + phone.officeCode() + phone.lineNumber());
        }
        if (factClass == gov.irs.factgraph.types.Day.class) {
            var day = (gov.irs.factgraph.types.Day) factValue;
            return new EvaluatedFactInfo(JAVA_STRING, day.date().toString());
        }
        if (factClass == gov.irs.factgraph.types.Enum.class) {
            if (StringUtils.isNotBlank(factValue.toString())) {
                return new EvaluatedFactInfo(JAVA_STRING, factValue.toString());
            }
            return null;
        }
        if (factClass == gov.irs.factgraph.types.CollectionItem.class) {
            var collectionItem = (gov.irs.factgraph.types.CollectionItem) factValue;
            return new EvaluatedFactInfo(JAVA_STRING, collectionItem.id().toString());
        }
        if (factClass == gov.irs.factgraph.types.Tin.class || factClass == gov.irs.factgraph.types.Ein.class) {
            return new EvaluatedFactInfo(JAVA_STRING, factValue.toString().replace("-", ""));
        }
        if (factClass == Boolean.class) {
            return new EvaluatedFactInfo(JAVA_BOOLEAN, factValue);
        }

        return new EvaluatedFactInfo(JAVA_STRING, factValue.toString());
    }

    private boolean suppressZeroDollarValue(final String factPath) {
        final String abstractPath = factPath.replaceAll(ABSTRACT_PATH_UUID_PATTERN, "*");
        return !exportZeroDollarFacts.contains(abstractPath);
    }
}
