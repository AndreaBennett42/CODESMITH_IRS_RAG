import { REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';
import {
  AUTHORIZATION_CODE_PARAM_NAME,
  determineRedirectUrl,
  REDIRECT_PARAM_NAME,
  SESSION_ID_PARAM_NAME,
} from './AuthorizeStateScreen.js';
import { v4 as uuidv4 } from 'uuid';

describe(`determineRedirectUrl function`, () => {
  const defaultRedirectUrl = `https://default.redirect.url`;
  const authCode = uuidv4();

  it(`properly creates the redirectUrl`, () => {
    // given
    const baseUrl = `https://default.redirect.url`;
    const redirectParam = null;
    const allowedRedirectUrls: string[] = [];
    const sessionIdParam = null;

    // when
    const result = determineRedirectUrl(baseUrl, redirectParam, allowedRedirectUrls, authCode, sessionIdParam);

    // then
    expect(result.protocol).toBe(`https:`);
    expect(result.host).toBe(`default.redirect.url`);
    expect(result.searchParams.size).toBe(2);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`appends the ${AUTHORIZATION_CODE_PARAM_NAME} to the default redirect Url`, () => {
    // given
    const redirectParam = null;
    const allowedRedirectUrls: string[] = [];
    const sessionIdParam = null;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`default.redirect.url`);
    expect(result.searchParams.size).toBe(2);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`sets the redirectUrl when ${REDIRECT_PARAM_NAME} is in the allowed list`, () => {
    // given
    const redirectParam = `https://custom.redirect.url`;
    const allowedRedirectUrls = [redirectParam];
    const sessionIdParam = null;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`custom.redirect.url`);
    expect(result.searchParams.size).toBe(2);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`uses the default Url the ${REDIRECT_PARAM_NAME} param when not in the allowed list`, () => {
    // given
    const redirectParam = `https://custom.redirect.url`;
    const allowedRedirectUrls = [`https://different.redirect.url`];
    const sessionIdParam = null;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`default.redirect.url`);
    expect(result.searchParams.size).toBe(2);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`uses the default Url the ${REDIRECT_PARAM_NAME} param when the allowed list is empty`, () => {
    // given
    const redirectParam = `https://custom.redirect.url`;
    const allowedRedirectUrls: string[] = [];
    const sessionIdParam = null;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`default.redirect.url`);
    expect(result.searchParams.size).toBe(2);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`appends the ${SESSION_ID_PARAM_NAME} as a queryParam`, () => {
    // given
    const redirectParam = null;
    const allowedRedirectUrls: string[] = [];
    const sessionIdParam = `123-abc-456`;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`default.redirect.url`);
    expect(result.searchParams.size).toBe(3);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(SESSION_ID_PARAM_NAME)).toBe(sessionIdParam);
    expect(result.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.AUTHSTATE);
  });

  it(`appends ${SESSION_ID_PARAM_NAME} and uses the ${REDIRECT_PARAM_NAME} param when in the allowed list`, () => {
    // given
    const redirectParam = `https://custom.redirect.url`;
    const allowedRedirectUrls = [`https://custom.redirect.url`];
    const sessionIdParam = `123-abc-456`;

    // when
    const result = determineRedirectUrl(
      defaultRedirectUrl,
      redirectParam,
      allowedRedirectUrls,
      authCode,
      sessionIdParam
    );

    // then
    expect(result.host).toBe(`custom.redirect.url`);
    expect(result.searchParams.size).toBe(3);
    expect(result.searchParams.get(AUTHORIZATION_CODE_PARAM_NAME)).toBe(authCode);
    expect(result.searchParams.get(SESSION_ID_PARAM_NAME)).toBe(sessionIdParam);
  });
});

describe(`AuthorizeStateScreen`, () => {
  it.todo(`Renders without error`, () => {});
});
