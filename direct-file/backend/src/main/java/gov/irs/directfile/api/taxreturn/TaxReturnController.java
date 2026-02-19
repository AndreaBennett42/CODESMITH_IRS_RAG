package gov.irs.directfile.api.taxreturn;

import java.lang.reflect.InvocationTargetException;
import java.net.*;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.modelmapper.record.RecordValueReader;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.config.IPAddressUtil;
import gov.irs.directfile.api.dataimport.DataImportService;
import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;
import gov.irs.directfile.api.errors.*;
import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.io.storagelocations.StorageLocationBuilder;
import gov.irs.directfile.api.pdf.PdfService;
import gov.irs.directfile.api.taxreturn.dto.*;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.user.UserService;
import gov.irs.directfile.api.user.domain.UserInfo;

@Profile("!mock")
@RestController
@Slf4j
@SuppressWarnings("PMD.CloseResource")
public class TaxReturnController implements TaxReturnApi {
    public static final String baseUrl = "/taxreturns";
    private static final ModelMapper modelMapper = new ModelMapper();
    private final TaxReturnService taxReturnService;
    private final UserService userService;
    private final PdfService pdfService;
    private final EncryptionCacheWarmingService cacheWarmingService;
    protected final DataImportService dataImportService;

    public TaxReturnController(
            TaxReturnService taxReturnService,
            UserService userService,
            PdfService pdfService,
            EncryptionCacheWarmingService cacheWarmingService,
            DataImportService dataImportService) {
        this.taxReturnService = taxReturnService;
        this.userService = userService;
        this.pdfService = pdfService;
        this.cacheWarmingService = cacheWarmingService;
        this.dataImportService = dataImportService;

        // Context:
        // https://github.com/modelmapper/modelmapper/issues/546#issuecomment-1068925341
        modelMapper.getConfiguration().addValueReader(new RecordValueReader());
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_ALL_BY_USERID)
    public ResponseEntity<List<ResponseBody>> getAllByUserId() {
        UserInfo userInfo = userService.getCurrentUserInfo();

        var dtos = taxReturnService.findByUserId(userInfo.id()).stream()
                .map(taxReturn -> mapToResponseBody(taxReturn))
                .collect(Collectors.toList());

        return ResponseEntity.ok(dtos);
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_BY_TAXRETURNID)
    public ResponseEntity<ResponseBody> getById(UUID id) {
        UserInfo userInfo = userService.getCurrentUserInfo();

        Optional<TaxReturn> taxReturn = taxReturnService.findByIdAndUserId(id, userInfo.id());

        if (taxReturn.isEmpty()) {
            throw new TaxReturnNotFoundResponseStatusException();
        }

        return ResponseEntity.ok(mapToResponseBody(taxReturn.get()));
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_CREATE)
    public ResponseEntity<ResponseBody> create(CreateRequestBody body, HttpServletRequest request) {
        UserInfo userInfo = userService.getCurrentUserInfo();

        cacheWarmingService.warmCacheForUserExternalId(userInfo.externalId());

        TaxReturn taxReturn = null;
        try {
            String remoteIpAddress = IPAddressUtil.getClientIpAddress(request);
            int remotePort = request.getRemotePort();
            final String userAgent = request.getHeader(HttpHeaders.USER_AGENT);

            taxReturn = taxReturnService.create(
                    body.getTaxYear(),
                    body.getFacts(),
                    userInfo.id(),
                    userInfo.email(),
                    userInfo.tin(),
                    remoteIpAddress,
                    remotePort,
                    userAgent);
        } catch (NonexistentDataException e) {
            throw new TaxReturnNotFoundResponseStatusException(e);
        } catch (InvalidOperationException e) {
            throw new ResponseStatusException(CreateResponseBadState.code, CreateResponseBadState.description, e);
        } catch (InvalidDataException
                | InvocationTargetException
                | IllegalAccessException
                | NoSuchMethodException
                | UnknownHostException e) {
            throw new ResponseStatusException(GenericResponseBadData.code, GenericResponseBadData.description, e);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        URI location = ServletUriComponentsBuilder.fromCurrentRequest()
                .path("/{id}")
                .buildAndExpand(taxReturn.getId())
                .toUri();

        // Asynchronously initialize the data-import request via virtual threads
        final UUID taxReturnId = taxReturn.getId();
        CompletableFuture.runAsync(() -> dataImportService.sendPreFetchRequest(
                taxReturnId, userInfo.id(), userInfo.externalId(), userInfo.tin(), body.getTaxYear()));

        return ResponseEntity.created(location).body(mapToResponseBody(taxReturn));
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_UPDATE)
    public ResponseEntity<Void> update(UUID id, UpdateRequestBody body, HttpServletRequest request) {
        UserInfo userInfo = userService.getCurrentUserInfo();
        cacheWarmingService.warmCacheForUserExternalId(userInfo.externalId());

        String referer = request.getHeader(HttpHeaders.REFERER);
        log.info("User {} is updating tax return {} from referer {}", userInfo.id(), id, referer);

        try {
            taxReturnService.update(id, body.getFacts(), body.getStore(), body.getSurveyOptIn(), userInfo.id());
        } catch (InvalidOperationException e) {
            throw new ResponseStatusException(ModifyResponseBadState.code, ModifyResponseBadState.description, e);
        } catch (InvalidDataException | InvocationTargetException | IllegalAccessException | NoSuchMethodException e) {
            throw new ResponseStatusException(GenericResponseBadData.code, GenericResponseBadData.description, e);
        }

        String location = ServletUriComponentsBuilder.fromCurrentRequest()
                .path("/{id}")
                .buildAndExpand(id)
                .toUri()
                .toString();

        return ResponseEntity.noContent().header(HttpHeaders.LOCATION, location).build();
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_SUBMIT)
    public ResponseEntity<String> submit(UUID id, SubmitRequestBody body, HttpServletRequest request) {
        UserInfo userInfo = userService.getCurrentUserInfo();

        cacheWarmingService.warmCacheForUserExternalId(userInfo.externalId());
        TaxReturn taxReturn;

        try {
            String remoteAddress = IPAddressUtil.getClientIpAddress(request);

            int remotePort = request.getRemotePort();
            final String userAgent = request.getHeader(HttpHeaders.USER_AGENT);

            taxReturn = taxReturnService.submit(
                    id, body.getFacts(), userInfo.id(), userInfo, remoteAddress, remotePort, userAgent);
        } catch (Exception e) {
            throw new ResponseStatusException(GenericResponseBadData.code, GenericResponseBadData.description, e);
        }

        return ResponseEntity.accepted()
                .body("Tax return "
                        + id
                        + " was dispatched to the electronic filing queue by user "
                        + userInfo.id()
                        + " at "
                        + taxReturn.getMostRecentSubmitTime());
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_SIGN)
    public ResponseEntity<String> sign(UUID id, SignRequestBody body, HttpServletRequest request) throws Exception {
        UserInfo userInfo = userService.getCurrentUserInfo();
        cacheWarmingService.warmCacheForUserExternalId(userInfo.externalId());
        String remoteAddress = IPAddressUtil.getClientIpAddress(request);
        int remotePort = request.getRemotePort();
        String userAgent = request.getHeader(HttpHeaders.USER_AGENT);

        taxReturnService.submit(id, body.facts(), userInfo.id(), userInfo, remoteAddress, remotePort, userAgent);

        return ResponseEntity.accepted().body("Signed request " + id + " was accepted");
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_STATUS)
    public StatusResponseBody status(UUID id) {
        UserInfo userInfo = userService.getCurrentUserInfo();
        return taxReturnService.getStatus(id, userInfo.id());
    }

    @Override
    @Auditable(event = EventId.PDF_READ)
    public ResponseEntity<InputStreamResource> pdf(UUID id, String languageCode) {
        UserInfo userInfo = userService.getCurrentUserInfo();
        var taxReturnOption = taxReturnService.findByIdAndUserId(id, userInfo.id());
        if (taxReturnOption.isEmpty()) {
            throw new TaxReturnNotFoundResponseStatusException();
        }
        TaxReturn taxReturn = taxReturnOption.get();
        String language = languageCode.trim().toLowerCase();

        try {
            var stream = pdfService.getTaxReturn(language, taxReturn, false);

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_PDF)
                    .header(
                            "Content-Disposition",
                            "attachment; filename="
                                    + StorageLocationBuilder.getTaxReturnDocumentFilename(
                                            "taxreturn", taxReturn.getTaxYear(), language))
                    .body(new InputStreamResource(stream));
        } catch (Exception e) {
            log.error("Error: " + e.getMessage(), e);
            throw new ResponseStatusException(
                    GetPdfResponseFailedToCreate.code, GetPdfResponseFailedToCreate.description, e);
        }
    }

    @Override
    @Auditable(event = EventId.TAX_RETURN_GET_POPULATED_DATA)
    public WrappedPopulatedData getPopulatedData(UUID id) {
        UserInfo userInfo = userService.getCurrentUserInfo();

        // To ensure that the `taxreturn_id` belongs to the current user.
        Optional<TaxReturn> taxReturn = taxReturnService.findByIdAndUserId(id, userInfo.id());
        if (taxReturn.isEmpty()) {
            throw new TaxReturnNotFoundResponseStatusException();
        }

        return dataImportService.getPopulatedData(
                id, userInfo.id(), taxReturn.get().getCreatedAt());
    }

    private ResponseBody mapToResponseBody(TaxReturn taxReturn) {
        var responseBodyDto = modelMapper.map(taxReturn, ResponseBody.class);
        UUID tr_id = taxReturn.getId();
        responseBodyDto.setIsEditable(taxReturnService.isTaxReturnEditable(tr_id));
        responseBodyDto.setDataImportBehavior(taxReturn.getDataImportBehavior());

        return responseBodyDto;
    }

    protected List<TaxReturnSubmissionResponseBody> mapToTaxReturnSubmissionListResponseBody(UUID taxReturnId) {
        List<TaxReturnSubmissionResponseBody> dtos = new ArrayList<>();
        List<TaxReturnSubmission> trs = taxReturnService.findTaxReturnSubmissionsForAPIResponse(taxReturnId);
        trs.forEach(x -> dtos.add(new TaxReturnSubmissionResponseBody(
                x.getId(), x.getCreatedAt(), x.getSubmitUserId(), x.getReceiptId(), x.getSubmissionReceivedAt())));
        return dtos;
    }

    // @PreDestroy
    // public void shutdownExecutor() {
    //     virtualThreadExecutor.shutdown();
    // }
}
