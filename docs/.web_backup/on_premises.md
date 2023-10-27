.. meta::
   :description: Documentation on how to deploy Konfuzio on premises using Kubernetes and Helm or Docker.

.. _Server Installation:

# Installation Guide

On-premises, also known as self-hosted, is a setup that allows Konfuzio Server to be implemented 100% on your own infrastructure or any cloud of your choice. In practice, it means that you know where your data is stored, how it's handled and who gets hold of it.

A common way to operate a production-ready and scalabe Konfuzio installation is via Kubernetens. An alternative deployment option is the [Single VM setup via Docker](/web/on_premises.html#docker-single-vm-setup). We recommend to use the option which is more familiar to you. In general

<div class="video-container">
    <iframe class="video" src="https://www.youtube.com/embed/KXc8FTU-2NM" allowfullscreen></iframe>
</div>

On-Premise Konfuzio installations allow to create Superuser accounts which can access all [Documents](https://help.konfuzio.com/modules/administration/superuserdocuments/index.html), [Projects](https://help.konfuzio.com/modules/administration/superuserprojects/index.html) and [AIs](https://help.konfuzio.com/modules/administration/superuserais/index.html) via a dedicated view as well as creating custom [Roles](https://help.konfuzio.com/modules/administration/superuserroles/index.html)

Konfuzio Server has beens successfully installed on various clouds.

Here you find a few examples:

- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)
- IBM Cloud
- Oracle Cloud Infrastructure (OCI)
- Alibaba Cloud
- VMware Cloud
- DigitalOcean
- Rackspace
- Salesforce Cloud
- OVH Cloud
- Hetzner
- Telekom Cloud


## Billing and License

When you purchase a Konfuzio Server self-hosted license online, we provide you with the necessary credentials to download the Konfuzio Docker Images. If you deploy via Kubernetes you can find the Helm Charts already [here](https://git.konfuzio.com/shared/charts). An essential part of this process is the unique BILLING_API_KEY. This key should be set as an environment variable when starting the Konfuzio Server using the following information.

### Setup Billing API 

The `BILLING_API_KEY` needs to be passed as environment variable to the running Docker container. 
This is inline with the fact that all configuration of Konfuzio is done via [environment variables](https://dev.konfuzio.com/web/on_premises.html#environment-variables-for-konfuzio-server).

Here is an example command to illustrate the setting of the `BILLING_API_KEY` environment variable:

```
docker run -e BILLING_API_KEY=your_api_key_here -d your_docker_image
```

In the command above, replace `your_api_key_here` with the actual billing API key and `your_docker_image` with the name of your Docker image.

Here's what each part of the command does:

-   `docker run`: This is the command to start a new Docker container.
-   `-e`: This option allows you to set environment variables. You can also use "--env" instead of "-e".
-   `BILLING_API_KEY=your_api_key_here`: This sets the "BILLING_API_KEY" environment variable to your actual API key.
-   `-d`: This option starts the Docker container in detached mode, which means it runs in the background.
-   `your_docker_image`: This is the name of your Docker image. You replace this with the actual name of your Docker image.


Instead of using ["plain" Docker](https://dev.konfuzio.com/web/on_premises.html#docker-single-vm-setup) we recommend to use our [Helm Chart](https://dev.konfuzio.com/web/on_premises.html#quick-start-via-kubernetes-and-helm) or [Docker-Compose](https://dev.konfuzio.com/web/on_premises.html#quick-start-via-docker-compose) to install Konfuzio Server. 

- For Docker-Compose you can simply replace the BILLING_API_KEY placeholder in the [docker-compose.yaml](https://dev.konfuzio.com/_static/docker-compose.yml) file.
- When using Helm on Kubernetes, the BILLING_API_KEY is set as 'envs.BILLING_API_KEY' in the [values.yaml](https://git.konfuzio.com/shared/charts/-/edit/master/values.yaml#L28) file.

The Konfuzio container continues to report usage to our billing server, i.e., app.konfuzio.com, once a day. We assure you that the containers do not transmit any customer data, such as the image or text that's being analyzed, to the billing server.

### Important Updates

If the unique key associated with the Contract on app.konfuzio.com is removed, your self-hosted installation will cease to function within 24 hours. Currently, there is no limit on the number of pages that can be processed within this time. The Konfuzio Server will also refuse to start if any Python file within the container has been modified to preserve the integrity of our license checks.

### Technical Background

Our server restarts itself after 500 requests by default. The code that prevents changes to Python files in the container has been obfuscated to ensure a significant challenge for anyone attempting to modify the source code.

### Air-gapped Environments and Upcoming Changes

For those operating the Konfuzio Server in air-gapped environments, your Konfuzio Docker images are licensed to operate for one year, based on the release date, without needing to connect to our billing server.

Soon, however, we will be introducing changes to our license check mechanism. The entire license check will be deactivated if a BILLING_API_KEY is not set. We are currently notifying self-hosted users about this upcoming change, urging them to configure their billing credentials. For more information, please refer to the Billing and License section in our documentation.

For users requiring further instructions, especially those operating in air-gapped environments, please [contact us](https://konfuzio.com/support) for assistance.

We appreciate your understanding and cooperation as we continue to enhance our security, licensing, and billing measures for all self-hosted installations of the Konfuzio Server.

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
Konfuzio. Options can be specified using Helm's `--set option.name=value` or `--values=my_values.yaml` command
line option. A complete list of command line options can be found [here](https://helm.sh/docs/helm/). This guide will
cover required values and common options.

Create a values.yaml file for your Konfuzio configuration. See Helm docs for information on how your values file will override the defaults.
Useful default values can be found in the [values.yaml](https://git.konfuzio.com/shared/charts/-/blob/master/values.yaml) in the chart [repository](https://git.konfuzio.com/shared/charts).

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

```
--set ingress.enabled=True
--set ingress.HOST_NAME=konfuzio.example.com
```

<!-- 
###### IPs

If you plan to use an automatic DNS registration service,you won't need any additional
configuration for Konfuzio, but you will need to deploy it to your cluster.

If you plan to manually configure your DNS records they should all point to a static IP.
For example if you choose `example.com` and you have a static IP of `10.10.10.10`, then
`konfuzio.example.com`, `registry.example.com` and `minio.example.com` (if using MinIO) should all resolve to
`10.10.10.10`.

_Include these options in your Helm install command:_

```
--set global.hosts.externalIP=10.10.10.
```
-->

##### Persistence

By default the setup will create Volume Claims with the expectation that a dynamic
provisioner will create the underlying Persistent Volumes. If you would like to customize
the storageClass or manually create and assign volumes,please review the [storage
documentation](https://kubernetes.io/docs/concepts/storage/volumes/).

**Important** : After initial installation, making changes to your storage settings requires manually editing Kubernetes
objects, so it's best to plan ahead before installing your production instance of Konfuzio to avoid extra storage
migration work.

##### TLS certificates

You should be running Konfuzio using https which requires TLS certificates. To get automated certificates using letesencrypt you need to install [cert-manager](https://cert-manager.io/docs/installation/helm/) in your cluster. If you
have your own wildcard certificate, you already have cert-manager installed, or you have
some other way of obtaining TLS certificates. For the default configuration, you must
specify an email address to register your TLS certificates.

_Include these options in your Helm install command:_

```
--set letsencrypt.enabled=True
--set letsencrypt.email=me@example.com
```

##### PostgreSQL

By default this Konfuzio provides an in-cluster PostgreSQL database, for trial purposes
only.

.. note::
  Unless you are an expert in managing a PostgreSQL database within a cluster, we do not recommended this configuration for use in production.**

- A single, non-resilient Deployment is used

You can read more about setting up your production-readydatabase in the PostgreSQL
documentation. As soon you have an external PostgreSQL database ready, Konfuzio can
be configured to use it as shown below.

_Include these options in your Helm install command:_

```
--set postgresql.install=false
--set global.psql.host=production.postgress.hostname.local
--set global.psql.password.secret=kubernetes_secret_name
--set global.psql.password.key=key_that_contains_postgres_password
```

##### Redis

All the Redis configuration settings are configured automatically.

##### Persistent Volume

Konfuzio relies on object storage for highly-available persistent data in Kubernetes. By default, Konfuzio uses a 
[persistent volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) within the cluster.

<!--
##### Outgoing email

By default outgoing email is disabled. To enable it,provide details for your SMTP server
using the `global.smtp` and `global.email` settings. You can find details for these
settings in the command line options.

`--set global.smtp.address=smtp.example.com:587`  
`--set global.smtp.AuthUser=username-here`  
`--set global.smtp.AuthPass=password-here`  
-->

##### CPU, GPU and RAM Resource Requirements

The resource requests, and number of replicas for the Konfuzio components in this
setup are set by default to be adequate for a small production deployment. This is
intended to fit in a cluster with at least 8 vCPU with AVX2 support enabled, 32 GB of
RAM and one Nvidia GPU with minimum 4GB which supports at least CUDA10.1 and CUDNN 7.0. If you are
trying to deploy a non-production instance, you can reduce the defaults in order to fit
into a smaller cluster. Konfuzio can work without a GPU. The GPU is used to train and run Categorization AIs. We observe a 5x faster training and a 2x faster execution on GPU compared to CPU.

##### Storage Requirements

This section outlines the initial storage requirements for the on-premises installation. It is important to take these
requirements into consideration when setting up your server, as the amount of storage needed may depend on the number of
documents being processed.

1. For testing purposes, a minimum of 10 GB is required per server (not per instance of a worker).
2. For serious use, a minimum of 100 GB should be directly available to the application. This amount should also cover
   the following:
    - Postgres, which typically uses 10% of this size.
    - Docker image storage, up to 25 GB should be reserved for upgrades.
3. Each page thumbnail adds 1-2 KB to the file size.
4. After uploading, the total file size of a page image and its thumbnails increases by approximately a factor of 3 (10
   MB becomes approximately 30 MB on the server).
5. To reduce storage usage, it is recommended to disable sandwich file generation by
   setting `ALWAYS_GENERATE_SANDWICH_PDF=False`.

#### Deploy using Helm

Once you have all of your configuration options collected, we can get any dependencies
and run Helm. In this example, we've named our Helm release `konfuzio`.

```
helm repo add konfuzio-repo https://git.konfuzio.com/api/v4/projects/106/packages/helm/stable
helm repo update
helm upgrade --install konfuzio konfuzio-repo/konfuzio-chart --values my_values.yaml
```

Please create a my_values.yaml file for your Konfuzio configuration. Useful default values can be found in the values.yaml in the chart repository. See Helm docs for information on how your values file will override the defaults. Alternativ you can specify you configuration using `--set option.name=value`. 

#### Monitoring the Deployment

The status of the deployment can be checked by running `helm status konfuzio` which
can also be done while the deployment is taking place if you run the command in
another terminal.

#### Autoscaling

The Konfuzio Server deployments can be scaled dynamically using a [Horizontal Pod Autoscaler](https://kubernetes.io/de/docs/tasks/run-application/horizontal-pod-autoscale/) and a [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler). The autoscaling configuration for the Konfuzio Server installation of https://app.konfuzio.com can be found in this [Helm Chart](https://git.konfuzio.com/shared/monitoring-charts).

#### Initial login

You can access the Konfuzio instance by visiting the domain specified during
installation. In order to create an initial superuser, please to connect to a running pod.

```
kubectl get pod
kubectl exec --stdin --tty my-konfuzio-* --  bash
python manage.py createsuperuser
```

### Quick Start via Kubernetes and Helm

The following commands allow you to get a Konfuzio Server installation running with minimal configuration effort and relying on the [default values](https://git.konfuzio.com/shared/charts/-/blob/master/values.yaml) of the Chart. This uses Postgres, Redis and S3 via [MinIO](https://min.io/) as in-cluster deployments. This setup is not suited for production and may use insecure defaults.

```
helm repo add konfuzio-repo https://git.konfuzio.com/api/v4/projects/106/packages/helm/stable
helm repo update
helm install my-konfuzio konfuzio-repo/konfuzio-chart  \  
  --set envs.HOST_NAME="host-name-for-you-installation.com"  \
  --set envs.BILLING_API_KEY="******"  \  
  --set image.tag="released-******"  \  
  --set image.imageCredentials.username="******"  \
  --set image.imageCredentials.password="******"
```

### Upgrade

Before upgrading your Konfuzio installation, you need to check the [changelog](./changelog_app.html)
corresponding to the specific release you want to upgrade to and look for any that might
pertain to the new version.

We also recommend that you take a backup first.

Upgrade Konfuzio following our standard procedure,with the following additions of:

1.  Check the change log for the specific version you would like to upgrade to
2.  Ensure that you have created a [PostgreSQL backup](https://www.postgresql.org/docs/11/backup.html) in the previous
    step. Without a backup, Konfuzio data might be lost if the upgrade fails.
3a. If you use a values.yaml, update image.tag="released-******" to the desired Konfuzio Server version.
```
helm install --upgrade my-konfuzio konfuzio-repo/konfuzio-chart -f values.yaml
```

3b. If you use "--set", you can directly set the desired Konfuzio Server version.
```
helm install --upgrade my-konfuzio konfuzio-repo/konfuzio-chart --reuse-values --set image.tag="released-******"
```

4. We will perform the migrations for the Database for PostgreSQL automatically.



## Docker - Single VM setup

Konfuzio can be configured to run on a single virtual machine, without relying on
Kubernetes. In this scenario, all necessary containers are started manually or with a
container orchestration tool of your choice.

### VM Requirements

We recommend a virtual machine with a minimum of 8 vCPU (incl. AVX2 support) and
32 GB of RAM and an installed Docker runtime. A Nvidia GPU is recommended but not
required. In this setup Konfuzio is running in the context of the Docker executor,
therefore there are no strict requirements for the VMs operating systems. However, we
recommend a Linux VM with Debian, Ubuntu, CentOS,or Redhat Linux.

### Login to the Konfuzio Docker Image Registry

We will provide you with the credentials to login into our Docker Image Registry, which allows you to access and download our Docker Images. This action requires an internet connection.

The internet connection can be turned off once the download is complete. In case there is
no internet connection available during setup, the container must be transferred with an
alternative method as a file to the virtual machine.

Registry URL: {PROVIDED_BY_KONFUZIO}  
Username: {PROVIDED_BY_KONFUZIO}  
Password: {PROVIDED_BY_KONFUZIO}  

```
docker login REGISTRY_URL
```
            
The Tag "latest" should be replaced with an actual version. A list of available tags can be found here: https://dev.konfuzio.com/web/changelog_app.html.

### Quick Start via Docker-Compose

The fastest method for deploying Konfuzio Server using Docker is through Docker-Compose. To get started, follow these steps:

- Install Docker-Compose in a version that is compatible with Compose file format 3.9.
- Run `docker compose version` to verify that Docker-Compose is available.
- Download the [docker-compose.yml](https://dev.konfuzio.com/_static/docker-compose.yml) file.
- Fill in the mandatory variables in the first section of the docker-compose file.
- Launch Konfuzio Server by running `docker compose up -d`.

If you prefer to use Docker exclusively, we provide detailed instructions for setting up the containers in each necessary step (Step 1-9).

### 1. Download Docker Image

After you have connected to the Registry, the Konfuzio docker image can be downloaded via “docker pull”.
```
docker pull REGISTRY_URL/konfuzio/text-annotation/master:latest
```

The Tag "latest" should be replaced with an actual version. A list of available tags can be found here: https://dev.konfuzio.com/web/changelog_app.html.

### 2. Setup PostgreSQL, Redis, BlobStorage/FileSystemStorage
The databases and credentials to access them are created in this step. Please choose your selected [databases](/web/on_premises.html#database-and-storage) at this point. You need to be able to connect to your PostgreSQL via psql and to your Redis server via redis-cli before continuing the installation.

In case you use FileSystemStorage and Docker volume mounts, you need to make sure the volume can be accessed by the konfuzio docker user (uid=999). You might want to run `chown 999:999 -R /konfuzio-vm/text-annotation/data` on the host VM.

The PostgreSQL database connection can be verified via psql and a connection string in the [following format](https://github.com/jazzband/dj-database-url#url-schema).
This connection string is later set as `DATABASE_URL`.
```
psql -H postgres://USER:PASSWORD@HOST:PORT/NAME
```

The Redis connection can be verified using redis-cli and a connection string. 
To use the connection string for `BROKER_URL`, `RESULT_BACKEND` and `DEFENDER_REDIS_URL` you need to append the database selector (e.g. redis://default:PASSWORD@HOST:PORT/0) 
```
redis-cli -u redis://default:PASSWORD@HOST:PORT
```

### 3. Setup the environment variable file
Copy the /code/.env.example file from the container and adapt it to your settings. The .env file can be saved anywhere on the host VM. In this example we use "/konfuzio-vm/text-annotation.env".

### 4. Init the database, create first superuser via cli and prefill e-mail templates
In this example we store the files on the host VM and mount the directory "/konfuzio-vm/text-annotation/data" into the container. In the first step we create a container with a shell to then start the initialization scripts within the container.
The container needs to be able to access IP addresses and hostnames used in the .env. This can be ensured using --add.host. In the example we make the host IP 10.0.0.1 available.

```
docker run -it --add-host=10.0.0.1 \
  --env-file /konfuzio-vm/text-annotation.env \
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \
  REGISTRY_URL/konfuzio/text-annotation/master:latest bash
```

```
python manage.py migrate
python manage.py createsuperuser
python manage.py init_email_templates
python manage.py init_user_permissions
```

After completing these steps you can exit and remove the container.

.. note::
  The username used during the createsuperuser dialog must have the format of a valid e-mail in order to be able to login later.

.. note::
  The default email templates can be customized by navigating to your on-prem installation in the section "Email Templates". [Section 11a](/web/on_premises.html#a-upgrade-to-newer-konfuzio-version), explains how these templates can be updated.

### 5. Start the container
In this example we start four containers. The first one to serve the Konfuzio web application. 

```
docker run -p 80:8000 --name web -d --add-host=host:10.0.0.1 \
  --env-file /konfuzio-vm/text-annotation.env \
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \
  REGISTRY_URL/konfuzio/text-annotation/master:latest
```

The second and third are used to process tasks in the background without blocking the web application. Depending on our load scenario, you might to start a large number of worker containers.

```
docker run --name worker1 -d --add-host=host:10.0.0.1 \  
  --env-file /konfuzio-vm/text-annotation.env \  
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \  
  REGISTRY_URL/konfuzio/text-annotation/master:latest \  
  celery -A app worker -l INFO --concurrency 1 -Q celery,priority_ocr,ocr,\  
  priority_extract,extract,processing,priority_local_ocr,local_ocr,\
  training,finalize,training_heavy,categorize  

docker run --name worker2 -d --add-host=host:10.0.0.1 \
  --env-file /konfuzio-vm/text-annotation.env \  
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \
  REGISTRY_URL/konfuzio/text-annotation/master:latest \
  celery -A app worker -l INFO --concurrency 1 -Q celery,priority_ocr,ocr,\
  priority_extract,extract,processing,priority_local_ocr,local_ocr,\
  training,finalize,training_heavy,categorize
```

The fourth container is a Beats-Worker that takes care of sceduled tasks (e.g. auto-deleted documents).

```
docker run --name beats -d --add-host=host:10.0.0.1 \  
  --env-file /konfuzio-vm/text-annotation.env \  
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \  
  REGISTRY_URL/konfuzio/text-annotation/master:latest \  
  celery -A app beat -l INFO -s /tmp/celerybeat-schedule
```

### [Optional] 6. Use Flower to monitor tasks

[Flower](https://flower.readthedocs.io/en/latest/screenshots.html) can be used a task monitoring tool. Flower will be only accessible for Konfuzio superusers and is part of the Konfuzio Server Docker Image.

```
docker run --name flower -d --add-host=host:10.0.0.1 \  
  --env-file /konfuzio-vm/text-annotation.env \  
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \
  REGISTRY_URL/konfuzio/text-annotation/master:latest \  
  celery -A app flower --url_prefix=flower --address=0.0.0.0 --port=5555
```

The Konfuzio Server application functions as a reverse proxy and serves the Flower application. In order for Django to correctly access the Flower application, it requires knowledge of the Flower URL. Specifically, the FLOWER_URL should be set to http://host:5555/flower.

`FLOWER_URL=http://host:5555/flower`

```mermaid
graph LR
subgraph Network
a("User")
end
subgraph Local Network / Cluster Network
a("User") --> e("Konfuzio Server")
e("Konfuzio Server") -- FLOWER_URL --> f("Flower")
end
```
Please ensure that the Flower container is not exposed externally, as it does not handle authentication and authorization itself.  

### [Optional] 7. Run Container for Email Integration

The ability to [upload documents via email](https://help.konfuzio.com/integrations/upload-by-email/index.html) can be achieved by starting a dedicated container with the respective environment variables.

```
SCAN_EMAIL_HOST = imap.example.com
SCAN_EMAIL_HOST_USER = user@example.com
SCAN_EMAIL_RECIPIENT = automation@example.com
SCAN_EMAIL_HOST_PASSWORD = xxxxxxxxxxxxxxxxxx
```

```
docker run --name flower -d --add-host=host:10.0.0.1 \  
  --env-file /konfuzio-vm/text-annotation.env \  
  --mount type=bind,source=/konfuzio-vm/text-annotation/data,target=/data \
  REGISTRY_URL/konfuzio/text-annotation/master:latest \  
  python manage.py scan_email
```

### [Optional] 8. Use Azure Read API (On-Premises or as Service)

The Konfuzio Server can work together with the [Azure Read API](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/overview-ocr). There are two options to use the Azure Read API in an on-premises setup.
1. Use the Azure Read API as a service from the public Azure cloud.
2. Install the Azure Read API container directly on your on-premises infrastructure via Docker.

The Azure Read API is in both cases connected to the Konfuzio Server via the following environment variables.
```
AZURE_OCR_KEY=123456789 # The Azure OCR API key  
AZURE_OCR_BASE_URL=http://host:5000 # The URL of the READ API  
AZURE_OCR_VERSION=v3.2 # The version of the READ API
```

For the first option, login into the Azure Portal and create a Computer Vision resource under the Cognitive Services section. 
After the resource is created the AZURE_OCR_KEY and AZURE_OCR_BASE_URL is displayed. Those need to be added as environment variable.

For the "on-premises" option, please refer to the [Azure Read API Container installation guide](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers?tabs=version-3-2).
Please fill in the [Microsoft Request Form](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/computer-vision-how-to-install-containers?tabs=version-3-2#request-approval-to-run-the-container) to get an API_KEY and ENDPOINT_URL which is compatible with [running the On-Premises container](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers?tabs=version-3-2). 
The ENDPOINT_URL must be reachable by the Azure Read API Container.   

#### How to fix common Azure Read API issues for On-Premises 

##### 1. Ensure AVX2 Compatibility:

Verify if your system supports Advanced Vector Extensions 2 (AVX2). You can do this by running the following command on Linux hosts:

```grep -q avx2 /proc/cpuinfo && echo AVX2 supported || echo No AVX2 support detected```

For further information, refer to [Microsoft's AVX2 support guide](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers#advanced-vector-extension-support).

##### 2. Verify Azure Read Container Status:

You can monitor the Azure Read Container's status through its web interface. 
To do this, access the container on Port 5000 (i.e. http://localhost:5000/status) from a web browser connected to the Container's network. 
You might need to replace 'localhost' with the IP/network name of the Azure container. 
This interface will also help detect issues such as invalid credentials or an inaccessible license server (possibly due to a firewall).
Refer to [Microsoft's guide on validating container status](https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/computer-vision-how-to-install-containers#validate-that-a-container-is-running) for further assistance.

##### 3. Consider Trying a Different Container Tag:

If your Azure Read API Container ceases to function after a restart, it may be due to an automatic upgrade to a new, potentially faulty Docker tag. 
In such cases, consider switching to a previous Docker tag. You can view the complete version history of Docker tags at the [Microsoft Artifact Registry](ttps://mcr.microsoft.com/product/azure-cognitive-services/vision/read/tags). 
Remember to uncheck the "Supported Tags Only" box to access all available versions.


### [Optional] 9. Install the Document Layout Analysis (Segmentation) Container

Download the container with the credentials provided by Konfuzio.

Registry URL: {PROVIDED_BY_KONFUZIO}  
Username: {PROVIDED_BY_KONFUZIO}  
Password: {PROVIDED_BY_KONFUZIO}  

```
docker login REGISTRY_URL  
docker pull REGISTRY_URL/konfuzio/text-annotation/detectron2/main:released-2023-08-03_11-39-42
docker run --env-file /path_to_env_file.env REGISTRY_URL/konfuzio/text-annotation/detectron2/main:released-2023-08-03_11-39-42 ./run.sh
```

The "Document Layout Analysis Container" needs to be started with the following environment variables which you can enter into your .env file
```
BROKER_URL=  # Set this to an unused Redis database (e.g. "redis://redis:6379/2")
RESULT_BACKEND=  # Set this to an unused Redis database (e.g. "redis://redis:6379/3")
```

We recommend to run the "Document Layout Analysis Container" with 8GB of RAM. The container can be started and used with less RAM, however this may not work on large images. 
After the "Document Layout Analysis Container" is running you need to set the [DETECTRON_URL](/web/on_premises.html#detectron-url) for Konfuzio Server to point to the "Document Layout Analysis Container".

To learn more about how the Document Layout Analysis Container works internally, please [click here](https://dev.konfuzio.com/web/on_premises.html#document-layout-analysis).

### [Optional] 10. Install document summarization container

Download the container with the credentials provided by Konfuzio

Registry URL: {PROVIDED_BY_KONFUZIO}  
Username: {PROVIDED_BY_KONFUZIO}  
Password: {PROVIDED_BY_KONFUZIO}  

```
docker login REGISTRY_URL
docker pull REGISTRY_URL/konfuzio/text-annotation/summarization/main:released-2022-11-29_21-55-57
docker run --env-file /path_to_env_file.env REGISTRY_URL/konfuzio/text-annotation/summarization/main:released-2022-11-29_21-55-57
```

The summarization container needs to be started with the following environment variables which you can enter into your .env file
```
GPU=True  # If GPU is present
TASK_ALWAYS_EAGER=False
C_FORCE_ROOT=True
BROKER_URL=  # Set this to an unused Redis database (e.g. "redis://redis:6379/4")
RESULT_BACKEND=  # Set this to an unused Redis database (e.g. "redis://redis:6379/5")

```

We recommend to run the segmentation container with 4GB of RAM. The segmentation container can be started and used with less RAM, however this may not work on large images. 
After the summarization container is running you need to set the [SUMMARIZATION_URL](/web/on_premises.html#summarization-url) to point to the segmentation container. 


### 11a. Upgrade to newer Konfuzio Version

1. To update Konfuzio to the latest released version, check the timestamped name of the latest release here: https://dev.konfuzio.com/web/changelog_app.html
2. SSH into the server that runs the Konfuzio Docker container
3. Open the `docker-compose.yml` file for editing, and search for the line:
```
image: git.konfuzio.com:5050/konfuzio/text-annotation/master:released-<timestamp>
```
4. Replace `released-<timestamp>` with the latest release you want to update to. For example, if the latest release happened on November 15, you will have something like `released-2023-11-15_09-39-24`, so you will change the image path to:
```
image: git.konfuzio.com:5050/konfuzio/text-annotation/master:released-2023-11-15_09-39-24
```
5. Run the following to have Docker pull the new image and rebuild the container:
```
docker compose up
```
6. Once the command completed successfully the server update is complete.

#### Updating existing email templates

During Konfuzio upgrades it is possible that we release some updates to the installed email-templates. It is important
to note, that these email templates, do not override any existing email templates. This ensures, that you can make
your own changes, without the risk of losing them during each upgrade. If you do want to update and sync your email
templates with each Konfuzio update. You can delete all the current email templates, or a single email template which needs 
updating, and then running `python manage.py init_email_templates` (see 4). This action will then set each template to
its default template value. 

### 11b. Downgrade to older Konfuzio Version

Konfuzio downgrades are performed by creating a fresh Konfuzio installation in which existing Projects can be imported.
The following steps need to be undertaken:
- Export the Projects that you want to have available after downgrade using [konfuzio_sdk](https://help.konfuzio.com/integrations/migration-between-konfuzio-server-instances/index.html#migrate-projects-between-konfuzio-server-instances). Please make sure you use a SDK version that is compatible with the Konfuzio Server version you want to migrate to.
- Create a new Postgres Database and a new Folder/Bucket for file storage which will be used for the downgraded version
- Install the desired Konfuzio Server version by starting with 1.)
- Import the projects using ["python manage.py project_import"](https://help.konfuzio.com/integrations/migration-between-konfuzio-server-instances/index.html#migrate-projects-between-konfuzio-server-instances)

### Load Scenario for Single VM with 32GB

#### Scenario 1: With self-hosted OCR

| Number of Container | Container Type | RAM | Capacity                                                                   |
| --- | --- | --- | --- |
| 1                   | Web Container          | 4GB | ...                                                                                  |
| 3                   | Generic Celery Worker  | 4GB | 1500 (3 x 500) Pages of Extraction, Categorization, or Splitting per hour            |
| 1                   | Self-Hosted OCR worker | 8GB | 1200 (1 y 1200) Pages) / hours (Not needed if external API Service is used)          |
| N                   | remaining Containers   | 4GB | ...                                                                                  |

With this setup, around 1200 Pages per hour can be processed using OCR and Extraction, around 750 Pages per hour can be processed if OCR, Categorization and Extraction are active, around 500 Pages per hour can be processed if OCR, Splitting, Categorization and Extraction are active.

#### Scenario 2: Without self-hosted OCR

| Number of Container | Container Type | RAM | Capacity                                                                  |
| --- | --- | --- |---------------------------------------------------------------------------|
| 1                   | Web Container          | 4GB | ...                                                                       |
| 5                   | Generic Celery Worker  | 4GB | 1500 (3 x 500) Pages of Extraction, Categorization, or Splitting per hour |
| N                   | remaining Containers   | 4GB | ...                                                                       |

With this setup, around 1500 Pages per hour can be processed using OCR and Extraction, around 750 Pages per hour can be processed if OCR, Categorization and Extraction are active, around 500 Pages per hour can be processed if OCR, Splitting, Categorization and Extraction are active.

.. note::
  In case you train large AI Models (>100 Training Pages) more than 4GB for Generic Celery Workers are needed. The Benchmark used an Extraction AI with "word" detection mode and 10 Labels in 1 Label Set. The capacity is shared between all Users using the Konfuzio Server setup.

  The level of parallelity of task processing and therefore throughput can be increased by the number of running "Generic Celery Workers". 

  The performance of any system that processes documents can be affected by various factors, including the number of simultaneous connections, the size of the documents being processed, and the available resources such as RAM and processing power.

  The number of simultaneous connections can also impact performance. If multiple users are uploading and processing large documents simultaneously, it can put a strain on the system's resources, leading to slower processing times.

  Other factors that can affect performance include network latency, disk speed, and the complexity of the processing algorithms. It's essential to consider all these factors when designing and deploying document processing systems to ensure optimal performance.

## Document Layout Analysis

The Konfuzio Server leverages Document Layout Analysis' capabilities to enhance its Natural Language Processing and Object Detection services. You can read more about it on our [blog](https://konfuzio.com/en/document-layout-analysis/). The platform's architecture features multiple Document Layout Analysis containers, each encapsulating a Fast API instance and a Worker. The modular structure of these containers allows scalability based on specific requirements.

Each Fast API instance and its corresponding Worker communicate with a shared Redis instance, a robust in-memory data structure store used for caching and message brokerage. As a high-performance web framework, Fast API is pivotal in constructing efficient APIs.

The process unfolds as follows: The Konfuzio Server forwards document layout analysis tasks to a Container via the Fast API instance. In response, the Fast API generates a task entry in Redis, which a Worker can subsequently pick up and process. Upon completion, the worker returns the results back to the Fast API via Redis, creating a seamless flow of information.

This architecture thus delivers a scalable, high-performance environment equipped to manage substantial loads. It ensures efficient object detection and natural language processing capabilities based on the detectron library, making it an optimal choice for handling complex computational tasks.

```mermaid
graph 
l("Konfuzio Server") <--> a
m("Other Applications") <--> a
a("Load-Balancer (Optional)") <--> b
a <--> d
a <--> f
b <--> t("Redis")
c <--> t("Redis")
d("Redis") <--> t("Redis")
e("Redis") <--> t("Redis")
f("Redis") <--> t("Redis")
g("Redis") <--> t("Redis")
subgraph all1["Detectron Container - 1"]
b("Fast API")
c("Detectron Worker") 
end
subgraph all2["Detectron Container - 2"]
d("Fast API")
e("Detectron Worker") 
end
subgraph all3["Detectron Container - N"]
f("Fast API")
g("Detectron Worker") 
end
```

Document Layout Analysis is available on [https://app.konfuzio.com](https://app.konfuzio.com) as well as for self-hosted environment via [Docker](https://dev.konfuzio.com/web/on_premises.html#optional-9-install-the-document-layout-analysis-segmentation-container) or on [Kubernetes via Helm](https://dev.konfuzio.com/web/on_premises.html#kubernetes).

## Docker-Compose vs. Kubernetes

When it comes to running the Konfuzio Server, the choice between Docker Compose and Kubernetes will depend on your specific requirements and use case.

Docker Compose can be a good choice if you are running Konfuzio Server on a single host in production or for testing and development purposes. With Docker Compose, you can define and customize the services required for the Konfuzio Server in a YAML file we provide, and then use a single command to start all the containers.

On the other hand, Kubernetes is more suitable for production environments where you need to run Konfuzio Server at scale, across multiple hosts, and with features such as auto-scaling and self-healing. Kubernetes has a steep learning curve, but it provides advanced features for managing and scaling containerized applications.

Regarding the use of Docker Compose in multiple VMs, while it's possible to use Docker Compose in a distributed environment, managing multiple VMs can be more work than using a dedicated orchestration platform like Kubernetes. Kubernetes provides built-in support for managing distributed systems and is designed to handle the complexities of running containers at scale.

In either case, you can use the same Docker image for the Konfuzio Server, which will ensure consistency and portability across different environments. Overall, the choice between Docker Compose and Kubernetes will depend on your specific needs, level of expertise, and infrastructure requirements.

## Custom AI model training via CI pipelines

Konfuzio is a platform that provides users with the ability to run custom AI workflows securely
through its Continuous Integration (CI) pipelines. These pipelines enable users to run
their code automatically and continuously, learning from human feedback on any errors or issues as soon as they occur.

In situations where the Kubernetes deployment option is not utilized, Konfuzio recommends using
a dedicated virtual machine to run these pipelines. This ensures that the pipelines are isolated
from other processes and are therefore less susceptible to interference or interruption.

To effectively run these CI pipelines, the selected CI application must support Docker and webhooks.
Docker allows for the creation and management of containers, which are used to package and distribute the
code to be tested. Webhooks, on the other hand, provide a way for the CI application to trigger the
pipeline automatically.

Furthermore, the CI application needs to have network access to the Konfuzio installation.
This is necessary to enable the CI application to interact with the Konfuzio platform, ensuring
that the CI pipelines are able to access the necessary resources and dependencies.

Overall, Konfuzio's use of CI pipelines provides a powerful tool for advanced users to run their AI workflows securely,
with the added benefit of automated testing and continuous feedback.

## SSO via OpenID Connect (OIDC) 

Konfuzio utilizes OpenID Connect (OIDC) for identity verification, implementing this through [Mozilla's Django OIDC](https://github.com/mozilla/mozilla-django-oidc).
OIDC is a well-established layer built atop the OAuth 2.0 protocol, enabling client applications to confirm the identity of end users. 
Numerous Identity Providers (IdP) exist that support the OIDC protocol, thus enabling the implementation of Single Sign-On (SSO) within your application. 
A commonly chosen provider is [Keycloak](https://www.keycloak.org/). In the subsequent section, you will find a tutorial on how to use Keycloak with OIDC.

### SSO Keycloak Integration

With Keycloak, you can implement the following workflows in Konfuzio:

- Use the Keycloak server to authenticate users with Konfuzio, automatically creating users in Konfuzio if they don't exist yet.
- Disable password login in Konfuzio, forcing users to use Keycloak to log in.
- Synchronize groups assigned to users in Keycloak with groups in Konfuzio.
- Use Keycloak to generate tokens that can be used to access the Konfuzio REST API.

Enabling the first workflow (authenticating Keycloak users with Konfuzio) is mandatory to support the other ones.

#### Set up

To start and set up Keycloak server:

1. Download the [Keycloak server](https://www.keycloak.org/downloads).
2. Install and start Keycloak server using [instructions](https://www.keycloak.org/docs/latest/server_installation/).
3. Open the Keycloak dashboard in browser (locally it's http://0.0.0.0:8080/). 
4. Create the admin user.
5. Login to the Administration Console.
6. Create a new Realm or use the existing one (Master).

.. image:: ./keycloak/add-realm.png

At this point you need to create a new Keycloak client to integrate it with Konfuzio.

1. Create a new Keycloak _client_ of type _OpenID Connect_. Give it any name.
2. When prompted for _client authentication_ or _access type_ (in older version) choose _On_ or _confidential_.
3. Make sure that the _direct access grants_ option is enabled.
4. Make sure that the _web origins_ URL list includes the URL where requests to Keycloak are coming from (i.e. your Konfuzio installation URL).
5. Move to the `Credentials` tab and save the `Secret` value. Store it somewhere as it will be needed later.

You can now create your users and groups from the relevant sections in the Keycloak dashboard.

.. warning:
  Usernames in Keycloak must be email addresses, otherwise the integration will not work.

#### Use the Keycloak server to authenticate users with Konfuzio

After the setup, you need to set the following environment variables for you Konfuzio Server installation:

- `KEYCLOAK_URL` (http://127.0.0.1:8080/ - for localhost)
- `OIDC_RP_SIGN_ALGO` (RS256 - by default)
- `OIDC_RP_CLIENT_ID` (client name the setup)
- `OIDC_RP_CLIENT_SECRET` (secret value from the setup)
- `SSO_ENABLED` (set to `True` to activate the integration)

After a restart, you should see an `SSO` button on the login page. Clicking it will redirect you to the Keycloak login page. After a successful login, you will be redirected back to Konfuzio and logged in.

.. warning:
  The Keycloak admin user cannot login into the Konfuzio server.

#### Disable password login in Konfuzio

Set the `PASSWORD_LOGIN_ENABLED` environment variable to `False` on your Konfuzio installation. This will remove the password login form from the login page, forcing users to use Keycloak to log in.

#### Synchronize groups assigned to users in Keycloak with groups in Konfuzio

Konfuzio can synchronize groups with Keycloak. This means that if you create a group in Keycloak and assign users to it, those users will be automatically added to an already existing group with the same name in Konfuzio. This is done by creating a _group mapper_ in Keycloak:

1. Select the client you created earlier.
2. Go to the _Client scopes_ tab inside the client.
3. You should see a scope which is the name of the client followed by `-dedicated`. Click on it.
4. Click _Configure a new mapper_.
5. Choose _Group Membership_.
6. In the settings for the mapper,
   1. Set the _Name_ to `groups` (lowercase).
   2. Set the _Token Claim Name_ to `groups` (same as above).
   3. Disable _Full group path_.
   4. Save.

You also need to set the `SYNCHRONIZE_KEYCLOAK_GROUPS_WITH_KONFUZIO_GROUPS` environment variable to `True` on your Konfuzio installation. For additional security, you can also set `OIDC_RENEW_ID_TOKEN_EXPIRY_SECONDS` (default: 15 minutes) to a lower value, so permissions are checked from the backend in shorter intervals (see [documentation](https://mozilla-django-oidc.readthedocs.io/en/stable/installation.html#validate-id-tokens-by-renewing-them)).

Now when assigning users to groups in Keycloak, the same groups will be created in Konfuzio (if they don't exist already) and the users will be added to them. Note that you still need to add permissions to the groups in Konfuzio for them to have any effect. You can automate this process with a CLI command, run `python manage.py import_groups_and_permissions --help` for more information.

An exception to this is a group created with the name `Superuser` on Keycloak. This group will map to the superuser status on Konfuzio, which gives access to everything on the instance. If you don't create a `Superuser` group on Keycloak, this will not be used.

#### Use Keycloak to generate tokens for the Konfuzio REST API

By default, the Konfuzio REST API requires [token authentication](https://dev.konfuzio.com/web/api-v3.html#token-authentication), which is only possible with regular username and password. However, your Keycloak server can be configured to directly generate tokens that are compatible with the REST API.

##### Set up the Keycloak client

You can reuse the client you've already created in the original client setup. However, by choosing the default *client authentication: on* option, users will also need to include the *secret* you've created to authenticate with the Keycloak server. This is not ideal for the REST API, so you probably want to create a new client with the *client authentication: off*/*public* option. If you're using groups synchronization, remember to add the `groups` mapper to the new client as well.

##### Get a token from Keycloak

Once you have your client setup, you can test the token generation. Create a request similar to this:

```
curl --request POST \
  --url https://sso.konfuzio.com/realms/myrealm/protocol/openid-connect/token \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data username=myaccount@konfuzio.com \
  --data password=mypassword \
  --data client_id=myclient \
  --data client_secret=mysecret \
  --data grant_type=password \
  --data scope=openid
```

Change the variables accordingly:

- `sso.konfuzio.com` should be replaced with your Keycloak installation URL.
- `myrealm` should be replaced with the name of your Keycloak realm.
- The `username` field should contain the username of the Keycloak user.
- The `password` field should contain the password of the Keycloak user.
- The `client_id` field should contain the name of the client you created earlier.
- The `client_secret` field is only necessary if you configured your client to be "confidential". If so, it should contain the client secret as shown in the _Credentials_ tab of the Keycloak client.

The request will return an `access_token` token field that you can use in the next step. Depending on your client setup, it will also include a `refresh_token` and various fields that determine how long the generated token(s) are valid (see the Keycloak documentation for more information).

##### Use the Keycloak token to access the REST API

With the token generated in the previous step, we can test a request to the Konfuzio API;

```
curl --request GET \
  --url https://app.konfuzio.com/api/v3/auth/me/ \
  --header 'Authorization: Bearer mytoken'
```

Change the variables accordingly:

- `app.konfuzio.com` should be replaced with your Konfuzio installation URL.
- `mytoken` should be replaced with the `access_token` token field returned by Keycloak.

If everything is configured correctly, the request should return information about the user whose credentials were used to generate the token:

```
{
	"username": "myaccount@konfuzio.com"
}
```

You can use the token to access any other endpoint of the Konfuzio REST API with the same user permissions as the user whose credentials were used to generate the token.

### SSO via other Identity Management software 
In order to connect your Identity Management software with Konfuzio using Single Sign-On (SSO), you'll need to follow these steps. Please note, this tutorial is meant for Identity Management software that supports OIDC (OpenID Connect), a widely adopted standard for user authentication that's compatible with many software platforms. If you're using Keycloak, please refer to the separate tutorial on connecting Konfuzio with Keycloak.

#### Step 1: Setting up your Identity Provider
This step varies depending on the Identity Management software you're using. You should refer to your Identity Management software's documentation for setting up OIDC (OpenID Connect). During the setup, you will create a new OIDC client for Konfuzio. You will typically need to provide a client name, redirect URI, and optionally a logo.

The redirect URI should be the URL of your Konfuzio instance followed by "/oidc/callback/", for example https://<your-konfuzio-instance>/oidc/callback/.

At the end of this step, you should have the following information:

- Client ID
- Client Secret
- OIDC Issuer URL

#### Step 2: Configuring mozilla-django-oidc
You'll need to set the following environment variables:

- `OIDC_RP_CLIENT_ID`: This should be the client ID from your Identity Management software.
- `OIDC_RP_CLIENT_SECRET`: This should be the client secret from your Identity Management software.
- `OIDC_OP_AUTHORIZATION_ENDPOINT`: This should be your OIDC issuer URL followed by "/authorize".
- `OIDC_OP_TOKEN_ENDPOINT`: This should be your OIDC issuer URL followed by "/token".
- `OIDC_OP_USER_ENDPOINT`: This should be your OIDC issuer URL followed by "/userinfo".
- `OIDC_OP_JWKS_ENDPOINT`: This should be your OIDC issuer URL followed by "/.well-known/jwks.json".
- `OIDC_RP_SIGN_ALGO`: This is the signing algorithm your Identity Management software uses. This is typically "RS256".
- `OIDC_RP_SCOPES`: These are the scopes to request. For basic authentication, this can just be "openid email".


#### Step 3: Restarting Konfuzio
After you have set these environment variables, you should restart your Konfuzio server so that the new settings can take effect. How you do this depends on how you have deployed Konfuzio, but it might involve restarting a Docker container or a Django server.

#### Step 4: Testing

Now you can go to your Konfuzio instance and you should be redirected to your Identity Management software login screen. 

## Migrate AIs and Projects

### Overview of Migration Methods

This table shows the two migration methods, "Project Export" and "AI File," for moving various elements Projects and Konfuzio Server environments. The table includes a list of elements such as Annotations, Categories, Label Sets, Labels, and more, and indicates which method is applicable for each element. The "Project Export" method is used for exporting whole Projects, while the "AI File" method is used for exporting elements that are specifically relevant for just running the AI on a different Konfuzio Server environments.

|                                                                                        | [Project Export](/web/on_premises.html#migrate-a-project) | [AI File](/web/on_premises.html#migrate-an-extraction-or-categorization-ai) | Evaluation |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------|------------|
| [Annotations](https://help.konfuzio.com/modules/annotations/index.html)                | Yes                                                       | No                                                                          | No         |
| [Categories](https://help.konfuzio.com/modules/categories/index.html)                  | Yes                                                       | No                                                                          | No         |
| [Categorization AI](https://help.konfuzio.com/modules/categorization/index.html)       | Yes                                                       | Yes                                                                         | No         |
| [Extraction AI](https://help.konfuzio.com/modules/extractions/index.html)              | Yes                                                       | Yes                                                                         | No         |
| [Splitting AI](https://help.konfuzio.com/modules/extractions/index.html)               | Yes                                                       | Yes                                                                         | No         |
| [All Documents](https://help.konfuzio.com/modules/documents/index.html)                | No                                                        | No                                                                          | No         |
| [Test-/Training Documents](https://help.konfuzio.com/modules/documents/index.html#id1) | Yes                                                       | No                                                                          | No         |
| [Label Sets](https://help.konfuzio.com/modules/sets/index.html)                        | Yes                                                       | No                                                                          | No         |
| [Labels](https://help.konfuzio.com/modules/labels/index.html)                          | Yes                                                       | No                                                                          | No         |
| [Members](https://help.konfuzio.com/modules/members/index.html)                        | No                                                        | No                                                                          | No         |


### Migrate an Extraction or Categorization AI

Superusers can migrate Extraction and Categorization AIs via the webinterface. This is explained on [https://help.konfuzio.com](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html). 

### Migrate a Project

Export the Project data from the source Konfuzio Server system.  
```
pip install konfuzio_sdk  
# The "init" command will ask you to enter a username, password and hostname to connect to Konfuzio Server.
# Please enter the hostname without trailing backslash. For example "https://app.konfuzio.com" instead of "https://app.konfuzio.com/".
# The init commands creates a .env in the current directory. To reset the SDK connection, you can delete this file.
konfuzio_sdk init
konfuzio_sdk export_project <PROJECT_ID>
```

The export will be saved in a folder with the name data_<project_id>. This folder needs to be transferred to the target system
The first argument is the path to the export folder, the second is the project name of the imported project on the target system.

```
python manage.py project_import "/konfuzio-target-system/data_123/" "NewProjectName"
```

Alternatively, you can merge the Project export into an existing Project.

```
python manage.py project_import "/konfuzio-target-system/data_123/" --merge_project_id <EXISTING_PROJECT_ID>
```

## Database and Storage

### Overview
To run Konfuzio Server, three types of storages are required. First, a PostgreSQL database is needed to store structured application data. Secondly, a storage for Blob needs to be present. Thirdly, a Redis database that manages the background Task of Konfuzio Server is needed. You can choose your preferred deployment option for each storage type and connect Konfuzio via environment variables to the respective storages. We recommend planning your storage choices before starting with the actual Konfuzio installation.

| Storage Name | Recommended Version | Supported Version        | Deployment Options |
| --- | --- |--------------------------| --- |
| [Postgres](https://www.postgresql.org/) | Latest Stable | PostgreSQL 12 and higher | Managed (Cloud) Service, VM Installation, Docker, In-Cluster* |
| [Redis](https://redis.io/) | Latest Stable | Redis 6 and higher      | Managed (Cloud) Service, VM Installation, Docker, In-Cluster* |
| Blob Storage | Latest Stable | All with activ support   | Filesystem, S3-compatible Service (e.g. [Amazon S3](https://aws.amazon.com/s3/), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs/)), In-Cluster* S3 via [MinIO](https://min.io/docs/minio/container/index.html) |

\*If you use [Kubernetes Deployment](/web/on_premises.html#kubernetes) you can choose the 'in-Cluster' option for Postgres, Redis and S3-Storage.

### Usage of PostgreSQL

Konfuzio Server will create a total of 43 tables and use the following data types. This information refers to release [2022-10-28-07-23-39](https://dev.konfuzio.com/web/changelog_app.html#released-2022-10-28-07-23-39).

| data_type | count |
| --- | --- |
| bigint                   |     6 |
| boolean                  |    35 |
| character varying        |    78 |
| date                     |     2 |
| double precision         |    29 |
| inet                     |     2 |
| integer                  |   138 |
| jsonb                    |    28 |
| smallint                 |     2 |
| text                     |    21 |
| timestamp with time zone |    56 |
| uuid                     |     1 |

<!-- This table was created by running select data_type, count(*) from information_schema.columns where table_schema = 'public'  group by data_type ; -->

## Environment Variables

### Environment Variables for Konfuzio Server

#### 1. General Settings

##### HOST_NAME
The HOSTNAME variable is used in the E-Mail templates: https://example.konfuzio.com or http://localhost:8000 for local development.

Note: Please include the protocol (e.g. http://) even the variable is named HOST_NAME.

_This is mandatory. Type: string_

##### ALLOWED_HOSTS
Default: HOST_NAME variable without the protocol.

A list of strings representing the host/domain names that Konfuzio can serve. 
This is a security measure to prevent HTTP Host header attacks, which are possible even under many seemingly-safe web server configurations.
See https://docs.djangoproject.com/en/4.2/ref/settings/#allowed-hosts

_Type: List[string]_

##### BILLING_API_KEY
The Billing API 
to connect with the Konfuzio License Server.  
See https://dev.konfuzio.com/web/on_premises.html#billing-and-license

_This is mandatory. Type: string_

##### DEBUG
Default: False

Use False for production and True for local development.
See https://docs.djangoproject.com/en/3.2/ref/settings/#std:setting-DEBUG

_Type: boolean_

##### SECRET_KEY
Insert random secret key.
See https://docs.djangoproject.com/en/4.0/ref/settings/#secret-key

_This is mandatory. Type: string_

##### DATABASE_URL
Please enter a Postgres Database in the format of "postgres://USER:PASSWORD@HOST:PORT/NAME" (https://github.com/kennethreitz/dj-database-url#url-schema).

_This is mandatory. Type: string_

##### DATABASE_URL_SSL_REQUIRE
Default: False

Specify whether the database connection requires SSL.

_Type: boolean_

##### MAINTENANCE_MODE
Default: False

Set maintenance mode, shows 503 error page when maintenance-mode is on.

_Type: boolean_

##### CSV_EXPORT_ROW_LIMIT
Default: 100

The (approximate) number of rows that an exported CSV file of extraction results can include.
Set to 0 for no limit, but beware of timeouts as the CSV files are exported synchronously.

_Type: integer_

##### ACCOUNT_BLOCK_FREE_EMAIL_PROVIDERS
Default: False

Block registration for users with an email address from a free provider according to
[the blocklist](https://github.com/Kikobeats/free-email-domains/blob/master/domains.json).

_Type: boolean_

##### DETECTRON_URL 
Default: None

This is used to connect to the optional [Document Layout Analysis Container](/web/on_premises.html#optional-9-install-document-segmentation-container). This is a URL in the form of 'http://detectron-service:8181/predict'.
You might need to adjust the detectron-service to your service name or IP.

_Type: string_

##### SUMMARIZATION_URL
Default: None

This is used to connect to the optional [summarization container](/web/on_premises.html#optional-10-install-document-summarization-container). This is a URL in the form of 'http://summarization-service:8181/predict'.
You might need to adjust the summarization-service to your service name or IP.

_Type: string_

#### 2. Background Tasks via Celery

Settings related to [Background Tasks](https://dev.konfuzio.com/web/explanations.html#background-processes).

##### BROKER_URL
Enter a Redis Database.
See https://docs.celeryq.dev/en/stable/userguide/configuration.html#std-setting-broker_url

_This is mandatory. Type: string_

##### RESULT_BACKEND
Enter a Redis Database.
See https://docs.celeryq.dev/en/stable/userguide/configuration.html#std-setting-result_backend

_This is mandatory. Type: string_

##### TASK_ALWAYS_EAGER
Default: False

When this is True, all background processes are executed in synchronously. 
See https://docs.celeryq.dev/en/stable/userguide/configuration.html#std-setting-task_always_eager

_Type: boolean_

##### WORKER_MAX_TASKS_PER_CHILD
Default: 50

After 50 tasks we replace a worker process to prevent and mitigate memory leaks.

_Type: integer_

##### WORKER_SEND_TASK_EVENTS
Default: False

See https://docs.celeryproject.org/en/stable/userguide/configuration.html#worker-send-task-events

_Type: boolean_

##### TASK_SEND_SENT_EVENT
Default: True

See https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-send-sent-event

_Type: boolean_

##### TASK_TRACK_STARTED
Default: True

See https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-track-started

_Type: boolean_

##### WORKER_MAX_MEMORY_PER_CHILD
Default: 100000

After allocation of 100MB we replace a worker process to mitigate memory leaks.

_Type: integer_

##### TASK_ACKS_ON_FAILURE_OR_TIMEOUT
Default: True

See https://docs.celeryproject.org/en/stable/userguide/configuration.html#std-setting-task_acks_on_failure_or_timeout

_Type: boolean_

##### TASK_ACKS_LATE
Default: True

See https://docs.celeryproject.org/en/stable/userguide/optimizing.html#reserve-one-task-at-a-time

_Type: boolean_

##### BROKER_MASTER_NAME
Default: None

The name of the Broker Master when using Redis Sentinel.
https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html

_Type: string_

##### RESULT_BACKEND_MASTER_NAME
Default: None

The name of the Result Backend Master when using Redis Sentinel.
https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html

_Type: string_

##### THRESHOLD_FOR_HEAVY_TRAINING
Default: 150

The Number of Document that indicate a heavy training. Heavy training are executed via a dedicated queue.
A "training_heavy" worker must be running in order to complete these trainings.

_Type: integer_

#### 3. OCR Settings

##### AZURE_OCR_KEY
Default: None

Get the API KEY from portal.azure.com for Microsoft.CognitiveServices.

_Type: string_

##### AZURE_OCR_BASE_URL
Default: None

Get the API URL from portal.azure.com for Microsoft.CognitiveServices

_Type: string_

##### AZURE_OCR_VERSION 
Default: 'v3.2'

Get the AZURE_OCR_VERSION from portal.azure.com for Microsoft.Cognitives.

_Type: string_

##### AZURE_OCR_MODEL_VERSION
Default='latest'

Get the AZURE_OCR_MODEL_VERSION from portal.azure.com for Microsoft.Cognitives.

_Type: string_

#### 4. Blob Storage Settings

##### DEFAULT_FILE_STORAGE
Default: 'django.core.files.storage.FileSystemStorage'

By default, the file system is used as file storage. 
To use an Azure Storage Account set 'storage.MultiAzureStorage'.
To use an S3-compatible storage set: 'storage.MultiS3Boto3Storage'.
See https://docs.djangoproject.com/en/4.0/ref/settings/#default-file-storage

_Type: string_

##### AZURE_ACCOUNT_KEY:
Default: ''

The Azure account key.

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiAzureStorage'. Type: string_

##### AZURE_ACCOUNT_NAME:
Default: ''

The name if the Azure account name.

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiAzureStorage'. Type: string_

##### AZURE_CONTAINER:
Default: ''

The name if the Azure storage container.

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiAzureStorage'. Type: string_

##### AWS_ACCESS_KEY_ID
Default: ''

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiS3Boto3Storage'. Type: string_

##### AWS_SECRET_ACCESS_KEY
Default: ''

The access key of the S3-Service.

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiS3Boto3Storage'. Type: string_

##### AWS_STORAGE_BUCKET_NAME
Default: ''

The bucket name of the S3-Service.

_This is mandatory if DEFAULT_FILE_STORAGE='storage.MultiS3Boto3Storage'. Type: string_

##### AWS_S3_REGION_NAME
Default: ''

The region name of the S3-Service.

_Type: string_

##### AWS_S3_ENDPOINT_URL
Default: ''

Custom S3 URL to use when connecting to S3, including scheme. 

_Type: string_

##### AWS_S3_USE_SSL
Default: True

Whether or not to use SSL when connecting to S3, this is passed to the boto3 session resource constructor.
See 

_Type: boolean_

##### AWS_S3_VERIFY
Default: None

Whether or not to verify the connection to S3. Can be set to False to not verify certificates or a path to a CA cert bundle.

_Type: string_

#### 5. Email Sending Settings

##### EMAIL_BACKEND
Default: django.core.mail.backends.smtp.EmailBackend
The SMTP backend for sending E-Mails.
See https://docs.djangoproject.com/en/4.0/ref/settings/#email-backend

_Type: string_

##### EMAIL_HOST
Default: ''

The host to use for sending email.

_Type: string_

##### EMAIL_HOST_PASSWORD
Default: '' 

Password to use for the SMTP server defined in EMAIL_HOST. This setting is used in conjunction with EMAIL_HOST_USER when authenticating to the SMTP server. If either of these settings is empty, we won’t attempt authentication.

_Type: string_

##### EMAIL_HOST_USER
Default: '' 

Username to use for the SMTP server defined in EMAIL_HOST. If empty, we won’t attempt authentication.

_Type: string_

##### EMAIL_PORT
Default: 25

Port to use for the SMTP server defined in EMAIL_HOST.
See https://docs.djangoproject.com/en/4.2/ref/settings/#std-setting-EMAIL_PORT

_Type: int_

##### EMAIL_USE_TLS
Default: False

See https://docs.djangoproject.com/en/4.0/ref/settings/#email-use-tls

_Type: boolean_

##### EMAIL_USE_SSL:
Default: False

See https://docs.djangoproject.com/en/4.0/ref/settings/#email-use-ssl

_Type: boolean_

##### DEFAULT_FROM_EMAIL
Default: 'support@konfuzio.net'

Specifies the email address to use as the "From" address when sending emails.

_Type: string_

##### SENDGRID_API_KEY
Default: None

In order to sent emails via sendgrid, set EMAIL_BACKEND='sendgrid_backend.SendgridBackend' and specify the Sendgrid API Key here.

_Type: string_

#### 6. Email Polling Settings

##### SCAN_EMAIL_HOST
Default: ''

The host to use for polling emails.

_Type: string_

##### SCAN_EMAIL_HOST_USER
Default: ''

Username to use for the SMTP server defined in SCAN_EMAIL_HOST.

_Type: string_

##### SCAN_EMAIL_RECIPIENT
Default: ''

Recipient email address to be polled. Emails sent to this address will be polled into Konfuzio.

_Type: string_

##### SCAN_EMAIL_HOST_PASSWORD
Default: ''

The password to use for the SMTP server defined in SCAN_EMAIL_HOST.

_Type: string_

##### SCAN_EMAIL_SLEEP_TIME
Default: 60

The polling interval in seconds.

_Type: int_

#### 7. Time limits for background tasks

More info about backgrounds tasks and their defaults can be viewed here: https://dev.konfuzio.com/web/explanations.html#celery-tasks.

##### EXTRACTION_TIME_LIMIT 
Default: 3600

The maximum extraction time in seconds.

_Type: int_

##### CATEGORIZATION_TIME_LIMIT 
Default: 180

The maximum categorization time in seconds.

_Type: int_

##### EVALUATION_TIME_LIMIT 
Default: 3600

The maximum evaluation time in seconds.

_Type: int_

##### TRAINING_EXTRACTION_TIME_LIMIT 
Default: 72000

The maximum extraction AI training time in seconds.

_Type: int_

##### TRAINING_CATEGORIZATION_TIME_LIMIT 
Default: 36000

The maximum categorization AI training time in seconds.

_Type: int_

##### SANDWICH_PDF_TIME_LIMIT 
Default: 1800

The maximum time to generate the sandwich PDF.

_Type: int_

##### DOCUMENT_TEXT_AND_BBOXES_TIME_LIMIT 
Default: 3600

The maximum time build a Document's text and bboxes.

_Type: int_

##### CLEAN_DELETED_DOCUMENT_TIME_LIMIT 
Default: 3600

The maximum time hard delete Documents, that are marked for deletion.
See https://help.konfuzio.com/modules/projects/index.html?#auto-deletion-of-documents

_Type: int_

##### CLEAN_DOCUMENT_WITHOUT_DATASET_TIME_LIMIT
Default: 3600

The maximum time to delete Documents which have reached the auto-deletion date.
See https://help.konfuzio.com/modules/projects/index.html?#auto-deletion-of-documents

_Type: int_

##### DOCUMENT_WORKFLOW_TIME_LIMIT
Default: 7200

The maximum time for the whole Document workflow in seconds. If a Document workflow does not complete within this time, the Document is set to an error state.

_Type: int_

<!--
#### Configure Azure OCR (optional).
AZURE_OCR_KEY=
AZURE_OCR_VERSION=
AZURE_OCR_BASE_URL=
-->

#### 8. Keycloak / SSO Settings

The following values establish a keycloak connection through the
mozilla oidc package (https://mozilla-django-oidc.readthedocs.io/en/stable/settings.html).

##### SSO_ENABLED
Default: True

Set to True active SSO login via Keycloak.

_Type: boolean_

##### KEYCLOAK_URL
Default: '127.0.0.1:8080/'

If you use keycloak version 17 and later set url like: http(s)://{keycloak_address}:{port}/ (optional).
If you use keycloak version 16 and earlier set url like: http(s)://{keycloak_address}:{port}/auth/ (optional).

_Type: string_

##### KEYCLOAK_REALM
Default: master

The Realm to be used.

_Type: string_

##### OIDC_RP_CLIENT_ID
Default: None

The OIDC Client ID. For Keycloak client creation see: https://www.keycloak.org/docs/latest/server_admin/#assembly-managing-clients_server_administration_guide (optional).

_Type: string_

##### OIDC_RP_CLIENT_SECRET
Default: None

The OIDC Client Secret. See https://mozilla-django-oidc.readthedocs.io/en/stable/settings.html#OIDC_RP_CLIENT_SECRET

_Type: string_

##### OIDC_RP_SIGN_ALGO
Default: RS256

Sets the algorithm the IdP uses to sign ID tokens.
See https://mozilla-django-oidc.readthedocs.io/en/stable/settings.html#OIDC_RP_SIGN_ALGO

_Type: string_

#### 9. Snapshot Settings

##### SNAPSHOT_DEFAULT_FILE_STORAGE
Default: 'django.core.files.storage.FileSystemStorage'

By default, the file system is used as file storage. 
To use an S3-compatible storage set: 'storages.backends.s3boto.S3BotoStorage'.
To use an Azure Blob Storage set: 'storages.backends.azure_storage.AzureStorage'.
See https://docs.djangoproject.com/en/4.0/ref/settings/#default-file-storage

_Type: string_

##### SNAPSHOT_LOCATION
Default: ''

The location where all Snapshots are saved (e.g. "/data/konfuzio-snapshots")

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='django.core.files.storage.FileSystemStorage'. Type: string_
_Type: string

##### SNAPSHOT_AWS_ACCESS_KEY_ID
Default: ''

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.s3boto.S3BotoStorage'. Type: string_

##### SNAPSHOT_AWS_SECRET_ACCESS_KEY
Default: ''

The access key of the S3-Service.

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.s3boto.S3BotoStorage'. Type: string_

##### SNAPSHOT_AWS_STORAGE_BUCKET_NAME
Default: ''

The bucket name of the S3-Service.

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.s3boto.S3BotoStorage'. Type: string_

##### SNAPSHOT_AWS_S3_REGION_NAME
Default: ''

The region name of the S3-Service.

_Type: string_

##### SNAPSHOT_AWS_S3_ENDPOINT_URL
Default: ''

Custom S3 URL to use when connecting to S3, including scheme. 

_Type: string_

##### SNAPSHOT_AWS_S3_USE_SSL
Default: True

Whether or not to use SSL when connecting to S3, this is passed to the boto3 session resource constructor.

_Type: boolean_

##### SNAPSHOT_AWS_S3_VERIFY
Default: None

Whether or not to verify the connection to S3. Can be set to False to not verify certificates or a path to a CA cert bundle.

_Type: string_

##### SNAPSHOT_AZURE_ACCOUNT_KEY:
Default: ''

The Azure account key.

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.azure_storage.AzureStorage'. Type: string_

##### SNAPSHOT_AZURE_ACCOUNT_NAME:
Default: ''

The name if the Azure account name.

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.azure_storage.AzureStorage'. Type: string_

##### SNAPSHOT_AZURE_CONTAINER:
Default: ''

The name if the Azure storage container.

_This is mandatory if SNAPSHOT_DEFAULT_FILE_STORAGE='storages.backends.azure_storage.AzureStorage'. Type: string_

##### SNAPSHOT_HOST_NAME
Default: None

This allows to set a different hostname which is used to make API calls when creating a Snapshot.  

_Type: string_

##### SNAPSHOT_RESTORE_ACROSS_ENVIRONMENTS
Default: False

Whether or not Snapshots from other environments can be restored.
 
_Type: boolean_


#### 10. Test Settings for Keycloak / SSO Settings

These variables are only used for Keycloak integration tests: 
The admin variables are for login keycloak admin panel, the test variables are for login to Konfuzio server.
The Keycloak Integration test can be run by installing Chromedriver and Selenium.

```
apt-get update && apt-get install -y fonts-liberation libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 libgbm1 libgtk-3-0 libxcomposite1 libxkbcommon0 xdg-utils libu2f-udev
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && apt-get install -y ./google-chrome-stable_current_amd64.deb
pip3 install coverage chromedriver-binary-auto chromedriver_binary
python manage.py test -t . server --tag=selenium-required
```

##### KEYCLOAK_ADMIN_USERNAME
Default: admin

The Keycloak admin username.

_Type: string_

##### KEYCLOAK_ADMIN_PASSWORD
Default: admin

The Keycloak admin password.

_Type: string_

##### KEYCLOAK_TEST_USERNAME
Default: fake@f.com

Username for Keycloak integration test.

_Type: string_

##### KEYCLOAK_TEST_PASSWORD
Default: pass1234

Password of User for Keycloak integration test.

_Type: string_


### Example .env file for Konfuzio Server

Konfuzio Server is fully configured via environment variables, these can be passed as dedicated environment variables or a single .env to the Konfuzio Server containers (REGISTRY_URL/konfuzio/text-annotation/master). A template for a .env file is provided here:

```text
# False for production, True for local development (mandatory).
# See https://docs.djangoproject.com/en/3.2/ref/settings/#std:setting-DEBUG
DEBUG=

# Set maintenance mode, shows 503 error page when maintenance-mode is on (mandatory).
MAINTENANCE_MODE=False

# Insert random secret key (mandatory).
# See https://docs.djangoproject.com/en/4.0/ref/settings/#secret-key
SECRET_KEY=

# The Billing API Key (optional) 
BILLING_API_KEY=
# The URL of the biling Server (optional)
BILLING_API_URL=https://app.konfuzio.com

# The HOSTNAME variable is used in the E-Mail templates (mandatory).
# https://example.konfuzio.com or http://localhost:8000 for local development.
# Note: Please include the protocol (e.g. http://) even the variable is named HOST_NAME
HOST_NAME=

# Please enter a Postgres Database (https://github.com/kennethreitz/dj-database-url#url-schema) (mandatory).
DATABASE_URL=
DATABASE_URL_SSL_REQUIRE=True

# Insert hostname e.g. konfuzio.com or * for local development (mandatory).
# See https://docs.djangoproject.com/en/4.0/ref/settings/#allowed-hosts
ALLOWED_HOSTS=

# Django's default storage (mandatory).
# for azure use: storage.MultiAzureStorage
# for S3-like storage use: storage.MultiS3Boto3Storage
# See https://docs.djangoproject.com/en/4.0/ref/settings/#default-file-storage
DEFAULT_FILE_STORAGE=django.core.files.storage.FileSystemStorage

# Required settings if storage.MultiAzureStorage is used (optional).
AZURE_ACCOUNT_KEY=
AZURE_ACCOUNT_NAME=
AZURE_CONTAINER=

# Required settings for storage.MultiS3Boto3Storage (optional).
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_STORAGE_BUCKET_NAME=
AWS_S3_REGION_NAME=
AWS_S3_ENDPOINT_URL=

# Access to customer Blob Storage (optional).
# e.g. "{'beispiel_ag': {'account_key': 'the_keys_124','azure_container': 'default',}}"
AZURE_CUSTOMER_STORAGE={}

# Celery settings (mandatory).
BROKER_URL=
RESULT_BACKEND=
TASK_ALWAYS_EAGER=True

# Defender (Brute-Force protection) settings (optional).
DEFENDER_REDIS_URL=

# SENTRY_DSN e.g. "https://123456789@sentry.io/1234567" (optional).
SENTRY_DSN=

# E-Mail address which is BCC in every transactional E-Mail (optional).
CRM_INTEGRATION_EMAIL=

# The SMTP credentials for sending E-Mails (optional).
# See https://docs.djangoproject.com/en/4.0/ref/settings/#email-backend
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=
EMAIL_HOST_PASSWORD=
EMAIL_HOST_USER=
EMAIL_PORT=25
# See https://docs.djangoproject.com/en/4.0/ref/settings/#email-use-tls
EMAIL_USE_TLS=False
# See https://docs.djangoproject.com/en/4.0/ref/settings/#email-use-ssl
EMAIL_USE_SSL=False
# See https://docs.djangoproject.com/en/4.0/ref/settings/#email-timeout
DEFAULT_FROM_EMAIL=

# Customize the email verification (optional)
# When set to “mandatory” the user is blocked from logging in until the email address is verified. Choose “optional” or “none” to allow logins with an unverified e-mail address. In case of “optional”, the e-mail verification mail is still sent, whereas in case of “none” no e-mail verification mails are sent.
ACCOUNT_EMAIL_VERIFICATION='mandatory'

# Api Key to sent emails via SendGrid if Debug=False (optional).
# If you use the SENDGRID_API_KEY you must also set EMAIL_BACKEND=sendgrid_backend.SendgridBackend
SENDGRID_API_KEY=

# Set Google Analytics or keep empty (optional).
GOOGLE_ANALYTICS=

# Captcha protected signup (optional).
CAPTCHA=False

# Flower URL or keep empty (optional).
FLOWER_URL=

# Rate limit per worker (optional).
TASK_DEFAULT_RATE_LIMIT=30/m

# Configure Azure OCR (optional).
AZURE_OCR_KEY=
AZURE_OCR_VERSION=
AZURE_OCR_BASE_URL=

# If this is activated SSL is required (optional).
# See https://docs.djangoproject.com/en/4.0/ref/settings/#std:setting-SESSION_COOKIE_SECURE
SESSION_COOKIE_SECURE=True

# If this is activated SSL is required (optional).
# https://docs.djangoproject.com/en/4.0/ref/settings/#csrf-cookie-secure
CSRF_COOKIE_SECURE=True

# New relic settings (optional).
NEW_RELIC_LICENSE_KEY=
NEW_RELIC_APP_NAME=
NEW_RELIC_ENVIRONMENT=

# Email integration (optional).
SCAN_EMAIL_HOST=
SCAN_EMAIL_HOST_USER=
SCAN_EMAIL_RECIPIENT=
SCAN_EMAIL_HOST_PASSWORD=

# Directory to cache files during the AI training process and when running AI models (optional).
KONFUZIO_CACHE_DIR =   # e.g. '/cache', uses tempdir if not set

# KEYCLOAK ENVIRONMENT The following values establish a keycloak connection through the
# mozilla oidc package (https://mozilla-django-oidc.readthedocs.io/en/stable/settings.html) (optional).

# SET TO TRUE TO ACTIVATE KEYCLOAK (optional).
SSO_ENABLED=True

# If you use keycloak version 17 and later set url like: http(s)://{keycloak_address}:{port}/ (optional).
# If you use keycloak version 16 and earlier set url like: http(s)://{keycloak_address}:{port}/auth/ (optional).
KEYCLOAK_URL=
KEYCLOAK_REALM=  # defaults to master

For Keycloak client creation see: https://www.keycloak.org/docs/latest/server_admin/#assembly-managing-clients_server_administration_guide (optional).
OIDC_RP_SIGN_ALGO=RS256
OIDC_RP_CLIENT_ID=
OIDC_RP_CLIENT_SECRET=

# These variables are only used for Keycloak integration tests: 
# The admin variables are for login keycloak admin panel, the test variables are for login to Konfuzio server (optional).
KEYCLOAK_ADMIN_USERNAME=
KEYCLOAK_ADMIN_PASSWORD=
KEYCLOAK_TEST_USERNAME=
KEYCLOAK_TEST_PASSWORD=

# Turn on/off autoretraining (optional).
TRAIN_EXTRACTION_AI_AUTOMATICALLY_IF_QUEUE_IS_EMPTY=False

# Turn on/off the immediate generation of sandwich pdf in full document workflow (optional).
ALWAYS_GENERATE_SANDWICH_PDF=True

# Default time limits for background tasks (optional).
# These defaults can be viewed here: https://dev.konfuzio.com/web/explanations.html#celery-tasks.
EXTRACTION_TIME_LIMIT = 
CATEGORIZATION_TIME_LIMIT = 
EVALUATION_TIME_LIMIT = 
TRAINING_EXTRACTION_TIME_LIMIT = 
TRAINING_CATEGORIZATION_TIME_LIMIT = 
SANDWICH_PDF_TIME_LIMIT = 
DOCUMENT_TEXT_AND_BBOXES_TIME_LIMIT = 
# Default time limits for period background tasks (optional)
# https://help.konfuzio.com/modules/projects/index.html?#auto-deletion-of-documents
# Both are set to 3600, the max amount of time the task may take. 
# If a huge amount of documents have been deleted, this may need to be increased. 
CLEAN_DELETED_DOCUMENT_TIME_LIMIT = 
CLEAN_DOCUMENT_WITHOUT_DATASET_TIME_LIMIT =

# Some models require a lot of ram during training. Defaults to 150 threshold which is the amount of training documents necessary to switch to a "training_heavy" celery queue
THRESHOLD_FOR_HEAVY_TRAINING = 
```

### Environment Variables for Read API Container
```text
 # The Azure OCR API key (mandatory).
AZURE_OCR_KEY=123456789
 # The URL of the READ API (mandatory).
AZURE_OCR_BASE_URL=http://host:5000
# The version of the READ API (optional).
AZURE_OCR_VERSION=v3.2
```

### Environment Variables for Document Layout Analysis Container

```text
# Connect Broker (mandatory).
BROKER_URL=
# Connect result backend (mandatory).
RESULT_BACKEND=
# Connect Sentry (optional).
SENTRY_DSN=
SENTRY_ENVIRONMENT=
SENTRY_RELEASE=
SENTRY_TRACES_SAMPLE_RATE=
# Setting for task processing (optional).
WORKER_SEND_TASK_EVENTS=
TASK_SEND_SENT_EVENT=
TASK_TRACK_STARTED=
TASK_ACKS_ON_FAILURE_OR_TIMEOUT=
TASK_ACKS_LATE=
```

## Profiling

[Django-silk](https://github.com/jazzband/django-silk) is an integration into Konfuzio Server which offers a detailed
overview of time spend in the database or within the internal code. When trying to troubleshoot performance issues,
wanting to understand where queries are made, or needing to cooperate with our support on technical problems, it may be
useful to use the Django-silk profiling to analyze performance. 

By default, the package is installed but turned off, you can enable the package by setting the following variable:

- `SILK_PROFILING: True`

By default, only 15% of requests are being profiled, this can be changed to any number between 0 and 100:

- `SILKY_INTERCEPT_PERCENT: 15`

To avoid the database from filling up, only the past 10k requests are being kept, this can be changed to any value:

- `SILKY_MAX_RECORDED_REQUESTS: 10000`

After enabling Django silk, the dashboard will become available on `http://localhost:8000/silk` for users who are 
logged in as superuser. The database can also be cleared by navigating to `http://localhost:8000/silk/cleardb/`. Do keep
in mind that this package does add some overhead querying to each request. So unless there are active problems, it
may be best to keep it turned off or at a low `SILKY_INTERCEPT_PERCENT` percentage. 
