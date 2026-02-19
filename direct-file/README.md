# direct-file

If you're ready to set up your local developer environment, go directly to [ONBOARDING.md](/ONBOARDING.md) and return back here for background information.

## Docker
First, some things that must be true for this to work:
* You have cloned this repository.
* You don't have other services occupying ports 3000, 8080, or 5432 (or have set alternate ports with environment variables as described below)

### Additional configuration for Apple M2 Laptop

If you are running docker on an Apple M2 laptop, you may also need to change the default file sharing implementation in your docker settings.

If you see the error `dependency failed to start: container direct-file-db is unhealthy` when attempting to start up the docker instance, try the following steps.

From the docker desktop:
* Enter the settings menu (click the gear icon in the top right)
* On the **General** tab, for `Choose file sharing implementation for your containers`
* Select the `gRPC FUSE` option ,
* Click the `Apply & Restart` button on the bottom right.

## Important configuration variables

| Name                    | Required | Default        | Description                                                                                                                             |
|-------------------------|----------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `DF_DB_USER_ID`         | No       | 999            | User id used to run the database (if you set this, it likely will be to the value of `id -u`.                                           |
| `DF_DB_GROUP_ID`        | No       | 999            | Group id used to run the database (if you set this, it likely will be to the value of `id -g`.                                          |
| `DF_DB_PORT`            | No       | 5432           | Port the backend api database will be exposed on outside of the docker network.                                                         |
| `MEF_APPS_DB_PORT`      | No       | 32768          | Port the submit/status database will be exposed on outside of the docker network                                                        |
| `STATEAPI_DB_PORT`      | No       | 5433           | Port the state api database will be exposed on outside of the docker network                                                            |
| `DF_CSPSIM_PORT`        | No       | 5000           | Port the CSP Simulator will be exposed on outside of the docker network.                                                                |
| `DF_EXTSVCSIM_PORT`     | No       | 5001           | Port the External Service Simulator (ESSAR, etc) will be exposed on outside of the docker network.                                      |
| `DF_API_PORT`           | No       | 8080           | Port the backend app is exposed to outside of docker.                                                                                   |
| `DF_STATUS_PORT`        | No       | 8082           | Port the status app is exposed to outside of docker.                                                                                    |
| `DF_SUBMIT_PORT`        | No       | 8083           | Port the submit app will run on and is exposed on outside of docker.                                                                    |
| `DF_FE_PORT`            | No       | 3000           | Port that will be exposed to access the frontend through docker.  This currently does not work for plain `npm start` outside of docker. |
| `DF_SCREENER_PORT`      | No       | 3500           | Port to access screener in docker.                                                                                                      |
| `DF_PROMETHEUS_PORT`    | No       | 9090           | Port that will be exposed to access the Prometheus dashboard from docker                                                                |
| `DF_GRAFANA_PORT`       | No       | 3030           | Port that will be exposed to access the Grafana dashboard from docker                                                                   |
| `DF_CLIENT_PUBLIC_PATH` | No       | `/df/file`     | Path prefix the client will be served from publicly.  This is embedded into build process and applies to docker builds.                 |
| `DF_API_PUBLIC_PATH`    | No       | `/df/file/api` | Path prefix the api will be served from publicly.                                                                                       |
| `MAVEN_OPTS`            | No       |                | Extra options to pass to maven, especially useful for setting a proxy.                                                                  |
| `DF_LISTEN_ADDRESS`     | No       | 127.0.0.1      | Listen address for docker services.  Set to "0.0.0.0" to listen on everything.                                                          |
| `DF_DISABLE_AUTO_LOGOUT`   | No        | false         | Disable autologout. This env var is only read by the node app through `VITE`, and only on `development`.

### Build

To build the factgraph, api, frontend, and setup a database simply run:

```bash
docker compose build
```

Then, to start it all:

```bash
docker compose up -d
```

Now you can use the application with a browser at http://localhost:5000 (or the port specified as `DF_CSPSIM_PORT`).  You will be accessing the authentication simulator directly, which will pass your traffic on to the client and backend api services in docker.

The backend api is at http://localhost:8080 (or `DF_API_PORT`) and the database is exposed on port 5432 (or `DF_DB_PORT`).

#### Common configurations

The default configuration if you run `docker compose up` will let you access the application in the browser through the authentication simulator at `DF_CSPSIM_PORT`.

Paths prefixed with `DF_CLIENT_PUBLIC_PATH` will be passed to the client and those prefixed with `DF_API_PUBLIC_PATH` will be passed to the API.  The most specific path match will be used, so these public prefixes may be nested.

Although the client and API services are behind authentication, the default configuration exposes their ports externally.  You can load the client in a browser directly, but API requests will fail because those requests would (due to client configuration) be through the CSP simulator and the browser would not have a valid cookie.

![Image of default docker compose development configuration](../docs/images/dev_config_docker_compose.png)

For client development, you can bypass the authentication simulator.  First, make sure the docker container for the client is not running (`docker compose rm -sf df-client`).    Next, start the client with `npm start` or `docker_dev_server.sh`.  Once it is started, access the client directly in the browser at `DF_CLIENT_PORT`.

![Image of common client development configuration](../docs/images/dev_config_client.png)

Other local configurations are possible.

### Monitoring

The applications use [OpenTelemetry](https://opentelemetry.io/docs/) locally for instrumenting observability metrics.

To enable and run the monitoring functionality locally, run:

```base
JAVA_TOOL_OPTIONS="-javaagent:/opentelemetry-javaagent.jar" docker compose --profile monitoring up -d --build
```

You can view what metrics we currently track through the Prometheus dashboard via http://localhost:9090 by default or `http://localhost:{DF_PROMETHEUS_PORT}` if `DF_PROMETHEUS_PORT` was set.

You can access and define dashboards through Grafana via http://localhost:3030 by default, or `http://localhost:{DF_GRAFANA_PORT}` if you've overridden the port. The default username is `admin` and the default password is `directfile`.

### Removing a service from docker compose

When you have a service running that you want to stop/remove, use:

```bash
docker compose rm --stop --force service-name
```

It can then be re-created and started with:

```bash
docker compose up -d service-name
```

### Git hooks

The maven Spotless plugin and the frontend `prettier` hook is used to help with standardizing formatting.  To check backend formatting of your current changes, run `./mvnw spotless:check`, and to apply those changes, use `./mvnw spotless:apply`.

#### Pre-commit

To make it easier to use, a `pre-commit` configuration has been added at the root of the repository.  You can install it with:

```bash
# linux
apt install pre-commit
# macos
brew install pre-commit

# and then, from a shell with cwd inside this repo:
pre-commit install
pre-commit install --hook-type pre-push
```

If you want to disable the checks, you can use:

```bash
pre-commit uninstall
```

or run your git commit with a `--no-verify` flag.

### Enable Optional Monitoring Service

Starts an optional OpenTelemetry collector, Prometheus, and Grafana instance for testing purposes.

```bash
JAVA_TOOL_OPTIONS="-javaagent:/opentelemetry-javaagent.jar" docker compose --profile=monitoring up -d --build
```

### Enable Debug of Containerized App
```bash
docker compose -f docker-compose.yaml -f docker-compose.debug.yaml up -d
```
In Visual Studio Code, add following to launch.json:
```
    {
        "type": "java",
        "name": "Attach to Remote Program",
        "request": "attach",
        "hostName": "localhost",
        "port": "5005",
        "projectName": "directfile-api"
    }
```
This will enable debugging for the backend project, and the same approach can be applied to other projects.

### Setup Redrive Policy for DLQ using CLI
Some CLI commands for reference to set redrive policy locally.
```
awslocal sqs list-queues (all DLQs prefixed by dlq-)

awslocal sqs get-queue-attributes --queue-url  <dlq-queue-url> --attribute-names QueueArn (get ARN for DQL)

awslocal sqs set-queue-attributes --queue-url <queue-url>  --attributes '{"RedrivePolicy":"{\"deadLetterTargetArn\":\"<dlq-queue-arn>\",\"maxReceiveCount\":\"2\"}"}'

awslocal sqs send-message --queue-url <source-queue-url> --message-body "Your message content"

awslocal sqs receive-message --queue-url <source-queue-url> --message-attribute-names All
```
