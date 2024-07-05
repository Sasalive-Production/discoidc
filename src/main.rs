use std::{
    collections::HashMap,
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime},
};

use anyhow::{bail, Context as _, Error};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{Response, StatusCode},
    response::{IntoResponse, Redirect},
    routing::get,
    Json, Router,
};
use chrono::Utc;
use clap::Parser;
use josekit::{
    jwe::{
        alg::aesgcmkw::{
            AesgcmkwJweAlgorithm::A128gcmkw, AesgcmkwJweDecrypter, AesgcmkwJweEncrypter,
        },
        JweHeader,
    },
    jwt::{self, JwtPayload},
};
use oauth2::{
    basic::BasicClient, reqwest::async_http_client, AuthorizationCode, ClientId, ClientSecret,
    CsrfToken, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, TokenResponse,
};
use openidconnect::{
    core::{
        CoreClaimName, CoreGenderClaim, CoreJsonWebKeySet, CoreJsonWebKeyType,
        CoreJweContentEncryptionAlgorithm, CoreJwsSigningAlgorithm, CoreProviderMetadata,
        CoreResponseType, CoreRsaPrivateSigningKey, CoreSubjectIdentifierType,
    },
    AdditionalClaims, Audience, AuthUrl, EmptyAdditionalProviderMetadata, EndUserEmail,
    EndUserName, EndUserUsername, IdToken, IdTokenClaims, IssuerUrl, JsonWebKeyId,
    JsonWebKeySetUrl, LocalizedClaim, PrivateSigningKey, ResponseTypes, Scope, StandardClaims,
    SubjectIdentifier, TokenUrl,
};
use rand::RngCore;
use rsa::{
    pkcs1::EncodeRsaPrivateKey,
    pkcs8::{der::zeroize::Zeroizing, LineEnding},
    RsaPrivateKey,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;
use url::Url;
use uuid::Uuid;

#[derive(Parser, Debug, Clone)]
struct Config {
    #[arg(long, env)]
    client_id: String,
    #[arg(long, env)]
    client_secret: String,
    #[arg(long, env)]
    issuer_url: Url,
    #[arg(long, env, default_value = "0.0.0.0:3000")]
    bind: SocketAddr,
    #[arg(long, env, default_value = "https://sasalivepro.com")]
    default_redirect: Url,
    #[arg(long, env)]
    key: Option<String>,
    #[arg(long, env, help = "client_id:redirect_uri")]
    clients: String,
    #[arg(
        long,
        env,
        help = "'guild_id:role_id:group_name' or 'guild_id:group_name'"
    )]
    mappings: String,
}

#[derive(Clone, Debug)]
struct Mapping {
    guild_id: String,
    role_id: Option<String>,
    group_name: String,
}

#[derive(Clone)]
struct AppState {
    client: BasicClient,
    clients: HashMap<String, Url>,
    mappings: Vec<Mapping>,
    config: Arc<Config>,
    rsa_pem: Zeroizing<String>,
    key_id: String,
    encrypter: AesgcmkwJweEncrypter,
    decrypter: AesgcmkwJweDecrypter,
}

#[derive(Serialize, Deserialize)]
struct Data {
    csrf_token: String,
    client_id: String,
    pkce_verifier: PkceCodeVerifier,
    state: Option<String>,
    scopes: Vec<AppScope>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AllowedResponseTypes {
    IdToken,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct AuthorizeParams {
    client_id: String,
    redirect_uri: Url,
    response_type: AllowedResponseTypes,
    scope: String,
    state: Option<String>,
    // code_challenge: String, TODO
    // code_challenge_method: String,
}

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum AppScope {
    OpenId,
    Profile,
    Email,
    Groups,
}

impl FromStr for AppScope {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "openid" => Ok(Self::OpenId),
            "profile" => Ok(Self::Profile),
            "email" => Ok(Self::Email),
            "groups" => Ok(Self::Groups),
            _ => bail!("invalid scope"),
        }
    }
}

async fn authorize(
    State(state): State<AppState>,
    Query(params): Query<AuthorizeParams>,
) -> Result<Response<Body>, AppError> {
    let scopes = params
        .scope
        .split(' ')
        .map(AppScope::from_str)
        .collect::<Result<Vec<_>, _>>()?;

    let Some(redirect_uri) = state.clients.get(&params.client_id) else {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "invalid_client",
            })),
        )
            .into_response());
    };

    if redirect_uri.clone() != params.redirect_uri {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "invalid_redirect_uri",
            })),
        )
            .into_response());
    };

    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

    let mut header = JweHeader::new();
    header.set_token_type("JWT");
    header.set_content_encryption("A128CBC-HS256");

    let current = SystemTime::now();
    let mut payload = JwtPayload::new();
    payload.set_claim(
        "data",
        Some(serde_json::to_value(Data {
            csrf_token: CsrfToken::new_random().secret().to_string(),
            pkce_verifier,
            client_id: params.client_id,
            state: params.state,
            scopes,
        })?),
    )?;
    payload.set_issued_at(&current);
    payload.set_expires_at(
        &current
            .checked_add(Duration::from_secs(60 * 10))
            .context("failed to add duration")?,
    );

    let jwt = jwt::encode_with_encrypter(&payload, &header, &state.encrypter)?;

    let (auth_url, _csrf_token) = state
        .client
        .authorize_url(|| CsrfToken::new(jwt))
        .add_scope(Scope::new("identify".to_string()))
        .add_scope(Scope::new("email".to_string()))
        .add_scope(Scope::new("guilds".to_string()))
        .add_scope(Scope::new("guilds.members.read".to_string()))
        .set_pkce_challenge(pkce_challenge)
        .url();

    Ok(Redirect::temporary(auth_url.as_ref()).into_response())
}

#[derive(Deserialize)]
struct User {
    id: String,
    username: String,
    global_name: String,
    email: String,
    verified: bool,
}

#[derive(Deserialize)]
struct GuildMember {
    roles: Vec<String>,
}

#[derive(Deserialize)]
struct CallbackParams {
    code: String,
    state: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct AppClaims {
    #[serde(skip_serializing_if = "Option::is_none")]
    groups: Option<Vec<String>>,
}
impl AdditionalClaims for AppClaims {}

type AppIdTokenClaims = IdTokenClaims<AppClaims, CoreGenderClaim>;

type AppIdToken = IdToken<
    AppClaims,
    CoreGenderClaim,
    CoreJweContentEncryptionAlgorithm,
    CoreJwsSigningAlgorithm,
    CoreJsonWebKeyType,
>;

async fn callback(
    State(state): State<AppState>,
    Query(params): Query<CallbackParams>,
) -> Result<Response<Body>, AppError> {
    let (payload, _) = jwt::decode_with_decrypter(params.state, &state.decrypter)?;

    if payload.expires_at().context("failed to get token exp")? < SystemTime::now() {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "token expired",
            })),
        )
            .into_response());
    }

    let data = serde_json::from_value::<Data>(
        payload
            .claim("data")
            .context("failed to get claim")?
            .clone(),
    )?;

    let redirect_uri = state
        .clients
        .get(&data.client_id)
        .context("invaild client")?;

    let token = state
        .client
        .exchange_code(AuthorizationCode::new(params.code))
        .set_pkce_verifier(data.pkce_verifier)
        .request_async(async_http_client)
        .await?;

    let user = reqwest::Client::new()
        .get("https://discord.com/api/users/@me")
        .header(
            "authorization",
            format!("Bearer {}", token.access_token().secret()),
        )
        .send()
        .await?
        .error_for_status()?
        .json::<User>()
        .await?;

    let mut guild_roles: HashMap<String, Vec<String>> = HashMap::new();

    let groups = if data.scopes.contains(&AppScope::Groups) {
        let mut groups = vec![];

        for m in state.mappings {
            let roles = match guild_roles.get(&m.guild_id) {
                Some(e) => Ok::<Vec<String>, Error>(e.to_vec()),
                None => {
                    let member: GuildMember = reqwest::Client::new()
                        .get(&format!(
                            "https://discord.com/api/users/@me/guilds/{}/member",
                            m.guild_id,
                        ))
                        .header(
                            "authorization",
                            format!("Bearer {}", token.access_token().secret()),
                        )
                        .send()
                        .await?
                        .error_for_status()?
                        .json::<GuildMember>()
                        .await?;

                    guild_roles.insert(m.guild_id.clone(), member.roles.clone());

                    Ok(member.roles)
                }
            }?;

            if let Some(role_id) = &m.role_id {
                if roles.contains(role_id) {
                    groups.push(m.group_name);
                }
            } else {
                groups.push(m.group_name);
            }
        }

        groups
    } else {
        vec![]
    };

    let mut claims = StandardClaims::new(SubjectIdentifier::new(user.id.clone()));
    if data.scopes.contains(&AppScope::Profile) {
        let mut username = LocalizedClaim::new();
        username.insert(None, EndUserName::new(user.global_name.clone()));

        claims = claims
            .set_name(Some(username))
            .set_preferred_username(Some(EndUserUsername::new(user.username)));
    }

    if data.scopes.contains(&AppScope::Email) {
        claims = claims
            .set_email(Some(EndUserEmail::new(user.email)))
            .set_email_verified(Some(user.verified));
    }

    let id_token = AppIdToken::new(
        AppIdTokenClaims::new(
            IssuerUrl::new(state.config.issuer_url.to_string())?,
            vec![Audience::new(data.client_id.to_string())],
            Utc::now() + chrono::Duration::seconds(86400),
            Utc::now(),
            claims,
            AppClaims {
                groups: data
                    .scopes
                    .contains(&AppScope::Groups)
                    .then(|| groups.clone()),
            },
        ),
        &CoreRsaPrivateSigningKey::from_pem(&state.rsa_pem, Some(JsonWebKeyId::new(state.key_id)))
            .map_err(|e| anyhow::anyhow!(e))?,
        CoreJwsSigningAlgorithm::RsaSsaPkcs1V15Sha256,
        None,
        None,
    )?;

    tracing::info!(
        "token issued: {:?} ({}) groups: {:?}",
        user.global_name,
        user.id,
        groups
    );

    let mut query: Vec<(&str, String)> = Vec::new();
    query.push(("id_token", id_token.to_string()));

    if let Some(state) = data.state {
        query.push(("state", state));
    }

    let url = Url::parse_with_params(redirect_uri.as_str(), query)?;

    Ok(Redirect::temporary(url.as_str()).into_response())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Arc::new(Config::parse());
    let clients = config
        .clients
        .split(',')
        .map(|s| {
            let parts = s.split(':').collect::<Vec<_>>();
            Ok::<(String, Url), Error>((parts[0].to_string(), Url::parse(&parts[1..].join(":"))?))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    let mappings = config
        .mappings
        .split(',')
        .map(|s| {
            let parts = s.split(':').collect::<Vec<_>>();
            let mapping = match parts.len() {
                2 => Mapping {
                    guild_id: parts[0].to_string(),
                    role_id: None,
                    group_name: parts[1].to_string(),
                },
                3 => Mapping {
                    guild_id: parts[0].to_string(),
                    role_id: Some(parts[1].to_string()),
                    group_name: parts[2].to_string(),
                },
                _ => bail!("invalid length"),
            };

            Ok::<Mapping, Error>(mapping)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let client = BasicClient::new(
        ClientId::new(config.client_id.clone()),
        Some(ClientSecret::new(config.client_secret.clone())),
        AuthUrl::new("https://discord.com/api/oauth2/authorize".to_string())?,
        Some(TokenUrl::new(
            "https://discord.com/api/oauth2/token".to_string(),
        )?),
    )
    .set_redirect_uri(RedirectUrl::from_url(
        config.issuer_url.clone().join("/callback")?,
    ));

    let metadata = CoreProviderMetadata::new(
        IssuerUrl::new(config.issuer_url.to_string())?,
        AuthUrl::new(config.issuer_url.join("/authorize")?.to_string())?,
        JsonWebKeySetUrl::new(config.issuer_url.join("/jwks")?.to_string())?,
        vec![
            ResponseTypes::new(vec![CoreResponseType::IdToken]), // ResponseTypes::new(vec![CoreResponseType::Code]),
                                                                 // ResponseTypes::new(vec![CoreResponseType::Token, CoreResponseType::IdToken]),
        ],
        vec![CoreSubjectIdentifierType::Public],
        vec![CoreJwsSigningAlgorithm::RsaSsaPssSha256],
        EmptyAdditionalProviderMetadata {},
    )
    .set_scopes_supported(Some(vec![
        Scope::new("openid".to_string()),
        Scope::new("email".to_string()),
        Scope::new("profile".to_string()),
        Scope::new("groups".to_string()),
    ]))
    .set_claims_supported(Some(vec![
        CoreClaimName::new("sub".to_string()),
        CoreClaimName::new("aud".to_string()),
        CoreClaimName::new("email".to_string()),
        CoreClaimName::new("email_verified".to_string()),
        CoreClaimName::new("exp".to_string()),
        CoreClaimName::new("iat".to_string()),
        CoreClaimName::new("iss".to_string()),
        CoreClaimName::new("name".to_string()),
        CoreClaimName::new("preferred_username".to_string()),
        CoreClaimName::new("groups".to_string()),
    ]));

    let mut rng = rand::thread_rng();
    let private = RsaPrivateKey::new(&mut rng, 2048)?;
    let rsa_pem = private.to_pkcs1_pem(LineEnding::LF)?;
    let key_id = Uuid::new_v4().to_string();

    let key = if let Some(key) = &config.key {
        key.as_bytes().to_owned()
    } else {
        let mut key = vec![0u8; 16];
        rng.try_fill_bytes(&mut key)?;

        key
    };

    let encrypter = A128gcmkw.encrypter_from_bytes(key.clone())?;
    let decrypter = A128gcmkw.decrypter_from_bytes(key)?;

    let state = AppState {
        client,
        clients,
        mappings,
        config: config.clone(),
        rsa_pem,
        key_id,
        encrypter,
        decrypter,
    };

    let jwks = CoreJsonWebKeySet::new(vec![CoreRsaPrivateSigningKey::from_pem(
        &state.rsa_pem,
        Some(JsonWebKeyId::new(state.key_id.clone())),
    )
    .unwrap()
    .as_verification_key()]);

    let app = Router::new()
        .route(
            "/",
            get(Redirect::temporary(config.default_redirect.as_ref())),
        )
        .route("/.well-known/openid-configuration", get(Json(metadata)))
        .route("/jwks", get(Json(jwks)))
        .route("/authorize", get(authorize))
        .route("/callback", get(callback))
        .with_state(state);
    let listener = TcpListener::bind(config.bind).await?;

    tracing::info!("Listening on: {}", config.bind);
    tracing::info!("Issuer URL: {}", config.issuer_url);
    axum::serve(listener, app).await?;

    Ok(())
}

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response<Body> {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("server error: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
