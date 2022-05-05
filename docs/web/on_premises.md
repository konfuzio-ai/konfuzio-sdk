.. meta::
   :description: Documentation on how to deploy Konfuzio on premises using Kubernetes and Helm.

# On-Premises Documentation

The recommended way to operate a production-ready and scalabe Konfuzio installation is via Kubernetes. An alternative and more light-weight deployment option is the [Single VM setup](/web/on_premises.html#alternative-deployment-options). 

## Kubernetes

### Required tools

Before deploying Konfuzio to your Kubernetes cluster, there are some tools you must have installed locally.

#### kubectl

kubectl is the tool that talks to the Kubernetes API. kubectl 1.15 or higher is required and it needs to be compatible 
with your cluster ([+/-1 minor release from your cluster](https://kubernetes.io/docs/tasks/tools/#before-you-begin)).

[> Install kubectl locally by following the Kubernetes documentation](https://kubernetes.io/docs/tasks/tools/#install-kubectl).

The server version of kubectl cannot be obtained until we connect to a cluster. Proceed
with setting up Helm.

#### Helm

Helm is the package manager for Kubernetes. Konfuzio is tested and supported with Helm v3.

##### Getting Helm

You can get Helm from the project's [releases page](https://github.com/helm/helm/releases), or follow other options 
under the official documentation of [installing Helm](https://helm.sh/docs/intro/install/).

###### Connect to a local Minikube cluster

For test purposes you can use `minikube` as your local cluster. If `kubectl cluster-info`
is not showing `minikube` as the current cluster, use `kubectl config set-cluster
minikube` to set the active cluster. For clusters in production please visit the [Kubernetes
Documentation](https://kubernetes.io/docs/setup/production-environment/).

##### Initializing Helm

If Helm v3 is being used, there no longer is an `init` sub command and the command is
ready to be used once it is installed. Otherwise please upgrade Helm.

#### Next steps

Once kubectl and Helm are configured, you can continue to configuring your Kubernetes
cluster.

### Deployment

Before running `helm install`, you need to make some decisions about how you will run
Konfuzio. Options can be specified using Helm's `--set option.name=value` command
line option. A complete list of command line options can be found [here](https://helm.sh/docs/helm/). This guide will
cover required values and common options.

#### Selecting configuration options

In each section collect the options that will be combined to use with `helm install`.

##### Secrets

There are some secrets that need to be created (e.g. SSH keys). By default they will be
generated automatically.

##### Networking and DNS

By default, Konfuzio relies on Kubernetes `Service` objects of `type: LoadBalancer` to
expose Konfuzio services using name-based virtual servers configured with `Ingress`
objects. You'll need to specify a domain which will contain records to resolve the
domain to the appropriate IP.

`--set global.hosts.domain=example.com`

###### IPs

If you plan to use an automatic DNS registration service,you won't need any additional
configuration for Konfuzio, but you will need to deploy it to your cluster.

If you plan to manually configure your DNS records they should all point to a static IP.
For example if you choose `example.com` and you have a static IP of `10.10.10.10`, then
`konfuzio.example.com`, `registry.example.com` and `minio.example.com` (if using MinIO) should all resolve to
`10.10.10.10`.

_Include these options in your Helm install command:_

`--set global.hosts.externalIP=10.10.10.`

##### Persistence

By default the setup will create Volume Claims with the expectation that a dynamic
provisioner will create the underlying Persistent Volumes. If you would like to customize
the storageClass or manually create and assign volumes,please review the [storage
documentation](https://kubernetes.io/docs/concepts/storage/volumes/).

**Important** : After initial installation, making changes to your storage settings requires manually editing Kubernetes
objects, so it's best to plan ahead before installing your production instance of Konfuzio to avoid extra storage
migration work.

##### TLS certificates

You should be running Konfuzio using https which requiresTLS certificates. By default, the setup will install and 
configure [cert-manager](https://github.com/jetstack/cert-manager) to obtain free TLS certificates. If you
have your own wildcard certificate, you already have cert-manager installed, or you have
some other way of obtaining TLS certificates. For the default configuration, you must
specify an email address to register your TLS certificates.

_Include these options in your Helm install command:_

`--set certmanager-issuer.email=me@example.com`

##### PostgreSQL

By default this Konfuzio provides an in-cluster PostgreSQL database, for trial purposes
only.

**NOTE: This configuration is not recommended for use in production.**

- A single, non-resilient Deployment is used

You can read more about setting up your production-readydatabase in the PostgreSQL
documentation. As soon you have an external PostgreSQLdatabase ready, Konfuzio can
be configured to use it as shown below.

_Include these options in your Helm install command:_

`--set postgresql.install=false`  
`--set global.psql.host=production.postgress.hostname.local`  
`--set global.psql.password.secret=kubernetes_secret_name`  
`--set global.psql.password.key=key_that_contains_postgres_password`  

##### Redis

All the Redis configuration settings are configured automatically.

##### Persistent Volume

Konfuzio relies on object storage for highly-available persistent data in Kubernetes. By default, Konfuzio uses a 
[persistent volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) within the cluster.

##### Outgoing email

By default outgoing email is disabled. To enable it,provide details for your SMTP server
using the `global.smtp` and `global.email` settings. You can find details for these
settings in the command line options.

`--set global.smtp.address=smtp.example.com:587`  
`--set global.smtp.AuthUser=username-here`  
`--set global.smtp.AuthPass=password-here`  

##### CPU, GPU and RAM Resource Requirements

The resource requests, and number of replicas for the Konfuzio components in this
setup are set by default to be adequate for a small production deployment. This is
intended to fit in a cluster with at least 16 vCPU with AVX2 support enabled, 32 GB of
RAM and one Nvidia GPU which supports at least CUDA10.1 and CUDNN 7.0. If you are
trying to deploy a non-production instance, you can reduce the defaults in order to fit
into a smaller cluster. Konfuzio can work withoutGPU, however runtime for extraction
and training of medium to large dataset will be significantly slower.

#### Deploy using Helm

Once you have all of your configuration options collected, we can get any dependencies
and run Helm. In this example, we've named our Helm release `konfuzio`.

`helm repo add konfuzio`  
`helm repo update`  
`helm upgrade --install konfuzio konfuzio/server \`  
`--timeout 600s \`  
`--set global.hosts.domain=example.com \`  
`--set global.hosts.externalIP=10.10.10.10 \`  
`--set certmanager-issuer.email=me@example.com`  

You can also use `--version <installation version>` option if you would like to install
a specific version of Konfuzio. This will output the list of resources installed once the
deployment finishes which may take 5-10 minutes.

#### Monitoring the Deployment

The status of the deployment can be checked by running `helm status konfuzio` which
can also be done while the deployment is taking place if you run the command in
another terminal.

#### Initial login

You can access the Konfuzio instance by visiting the domain specified during
installation. If you manually created the secret for initial root password, you can use that
to sign in as `root` user. If not, Konfuzio would've automatically created a random
password for `root` user. This can be extracted by the following command (replace
`<name>` by name of the release - which is `konfuzio` if you used the command above).

`kubectl get secret <name>-konfuzio-initial-root-password`  
`-ojsonpath='{.data.password}' | base64 --decode ; echo`  

### Upgrade

Before upgrading your Konfuzio installation, you need to check the [changelog](./changelog_app.html)
corresponding to the specific release you want to upgrade to and look for any that might
pertain to the new version.

We also recommend that you take a backup first.

Upgrade Konfuzio following our standard procedure,with the following additions of:

1. Check the change log for the specific version you would like to upgrade to
2. Ensure that you have created a [PostgreSQL backup](https://www.postgresql.org/docs/11/backup.html) in the previous
   step. Without a backup, Konfuzio data might be lost if the upgrade fails.
3. Go through deployment section step by step
4. Extract your previous `--set` arguments with (see Action1)
5. Decide on all the values you need to set
6. Perform the upgrade, with all `--set` arguments extracted(see Action 2)
7. We will perform the migrations for the Database forPostgreSQL automatically.

_Action 1_

`helm get values konfuzio > konfuzio.yaml`

_Action 2_

`helm upgrade kofuzio \`  
`--version <new version> \`  
`-f konfuzio.yaml \`  
`--set konfuzio.migrations.enabled=true \`  
`--set ...`  

## Docker

### Single VM setup

Konfuzio can be configured to run on a single virtual machine, without relying on
Kubernetes. In this scenario, all necessary containers are started manually or with a
container orchestration tool of your choice.

We recommend a virtual machine with a minimum of 12 vCPU (incl. AVX2 support) and
64GB of RAM and an installed Docker runtime. A Nvidia GPU is recommended but not
required. In this setup Konfuzio is running in the context of the Docker executor,
therefore there are no strict requirements for the VMs operating systems. However, we
recommend a Linux VM with Debian, Ubuntu, CentOS,or Redhat Linux.

#### 1. Download Docker Image

The Konfuzio docker image can be downloaded via “docker pull”. We will provide you
with the credentials. This action requires an internet connection.

The internet connection can be turned off once the download is complete. In case there is
no internet connection available during setup, the container must be transferred with an
alternative method as a file to the virtual machine.

Registry URL: registry.gitlab.com  
Username: {PROVIDED_BY_KONFUZIO}  
Password: {PROVIDED_BY_KONFUZIO}  

`> docker login registry.gitlab.com`  
`> docker pull registry.gitlab.com/konfuzio/text-annotation/master:latest`  

The Tag "latest" should be replaced with an actual version. A list of available tags can be found here: https://dev.konfuzio.com/web/changelog_app.html.

#### 2. Setup PostgreSQL, Redis, BlobStorage/FileSystemStorage
The database credentials are needed in the next step. You may want to use psql and redis-cli to check if database credentials are working.

#### 3. Setup environment variable file
Copy the /code/.env.example file from the container and adapt it to your settings. The .env file can be saved anywhere on the host VM. In this example we use "/konfuzio-vm/text-annotation.env".

#### 4. Init the database, create first superuser via cli and prefill e-mail templates
In this example we store the files on the host VM and mount the directory "/konfuzio-vm/text-annotation/data" into the container. In the first step we create a container with a shell to then start the initialization scripts within the container.
The container needs to be able to access IP addresses and hostnames used in the .env. This can be ensured using --add.host. In the example we make the host IP 10.0.0.1 available.

docker run -it --add-host=10.0.0.1 --env-file /konfuzio-vm/text-annotation.env --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data registry.gitlab.com/konfuzio/text-annotation/master:latest bash

`python manage.py migrate`  
`python manage.py createsuperuser`  
`python manage.py init_email_templates`  
`python manage.py init_user_permissions`  

After completing these steps you can exit and remove the container.

Note: The username used during the createsuperuser dialog must have the format of a valid e-mail in order to be able to login later.

#### 5. Start the container
In this example we start three containers, the first one to serve the Konfuzio web application. The second and third are used to process tasks in the background without blocking the web application.

`docker run -p 80:8000 --name web -d --add-host=host:10.0.0.1 \`  
`--env-file /konfuzio-vm/text-annotation.env \`  
`--mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \`  
`registry.gitlab.com/konfuzio/text-annotation/master:latest`

`docker run --name worker1 -d --add-host=host:10.0.0.1 \`  
`--env-file /konfuzio-vm/text-annotation.env \`  
`--mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \`  
`registry.gitlab.com/konfuzio/text-annotation/master:latest \`  
`celery -A app worker -l INFO --concurrency 1 -Q celery,priority_ocr,ocr,\`  
`priority_extract,extract,processing,priority_local_ocr,local_ocr,training,finalize,training_heavy,categorize`

`docker run --name worker2 -d --add-host=host:10.0.0.1 \`  
`--env-file /konfuzio-vm/text-annotation.env \`  
`--mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \`  
`registry.gitlab.com/konfuzio/text-annotation/master:latest \`  
`celery -A app worker -l INFO --concurrency 1 -Q celery,priority_ocr,ocr,\`  
`priority_extract,extract,processing,priority_local_ocr,local_ocr,training,finalize,training_heavy,categorize`

#### [Optional] 6. Use Flower to monitor tasks

[Flower](https://flower.readthedocs.io/en/latest/screenshots.html) can be used a task monitoring tool. Flower will be only accessible for Konfuzio superusers.

```
docker pull mher/flower:0.9.7 
docker run --rm --name flower -d -p 5555:5555  mher/flower:0.9.7 --adress 0.0.0.0 --url_prefix=flower --broker=redis://:@10.0.0.1:6379/0
```

The Konfuzio Server application acts as a reverse proxy an servers the flower application. Therefore, django needs to know the flower url. `FLOWER_URL=http://host:5555/flower`

#### [Optional] 7. Use Azure Read API on-premise
The [Azure Read API](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers?tabs=version-3-2) can be installed on-premise and used togehter with Konfuzio.

Please install the Read API Container according to the current [manual](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers?tabs=version-3-2)

Once the Azure Read API container is running you need to set the following variables in the .env file. This for example look like the following:

`AZURE_OCR_KEY=123456789 # The Azure OCR API key`  
`AZURE_OCR_BASE_URL=http://host:5000 # The URL of the READ API`  
`AZURE_OCR_VERSION=v3.2 # The version of the READ API`

#### [Optional] 8. Install document segmentation container

Download the container with the credentials provided by Konfuzio

Registry URL: registry.gitlab.com  
Username: {PROVIDED_BY_KONFUZIO}  
Password: {PROVIDED_BY_KONFUZIO}  

`> docker login registry.gitlab.com`  
`> docker pull registry.gitlab.com/konfuzio/detectron2:2021-10-07_13-45-34`  
`> docker run --env-file /path_to_env_file.env registry.gitlab.com/konfuzio/detectron2:2021-10-07_13-45-34 bash -c "export LC_ALL=C.UTF-8; export LANG=C.UTF-8; ./run_celery.sh"`

The segmentation container needs to be started with the following environment variables which you can enter into your .env file
```
GPU=True  # If GPU is present
C_FORCE_ROOT=True
BROKER_URL=  # See the konfuzio container
RESULT_BACKEND=  # See the konfuzio container
SENTRY_ENVIRONMENT=  # Optional
SENTRY_RELEASE=  # Optional
SENTRY_DSN=  # Optional
```

#### 9. Upgrade to newer Konfuzio Version

Konfuzio upgrades are performed by replacing the Docker Tag to the [desired version](https://dev.konfuzio.com/web/changelog_app.html)
After starting the new Containers Database migrations need to be applied by `python manage.py migrate` (see 4.).
In case additional migration steps are needed, they will be mentioned in the release notes.

## Alternative deployment options

### Custom AI model training via CI pipelines

Konfuzio uses CI pipelines to allow users to run customAI model code securely. In case
the Kubernetes deployment option is not used, we recommend a dedicated virtual
machine to run these pipelines. The selected CI application needs to support Docker
and webhooks. The CI application needs network access to the Konfuzio installation.
